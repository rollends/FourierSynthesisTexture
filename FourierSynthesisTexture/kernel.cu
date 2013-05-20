#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "cufft.lib")

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <Windows.h>
#include <GdiPlus.h>
#include <sstream>
using namespace Gdiplus;

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "cufft.h"
#include "device_launch_parameters.h"

const dim3 KERNEL_THREAD_COUNT(16, 16);

dim3 CreateGrid(size_t threadXCount, size_t threadYCount);
__global__ void ExecuteFilterSymmetric(cuComplex* freqDomain, size_t width, size_t height, size_t trueHeight, float roughness);
__global__ void GetMagnitudeSymmetric(cuComplex* input, float* output, size_t width, size_t height, size_t trueHeight);
__global__ void CreateRandom2DTexture(float* texture, size_t width, size_t height, unsigned long long randomSeed);

cudaError_t CreateFourierTextureFast(float* hostTexture, size_t texWidth, size_t texHeight, float* timeMSPerGeneration, float* totalTimeMS, size_t* usedBytes, int stepMax);
void printCudaError(const char * const message, cudaError_t error);
int getEncoderClsid(const WCHAR* format, CLSID* pClsid);

float ScaledNormalizeValue(float input, float min, float max)
{
	float norm = (input - min)/(max - min);
	return sqrt(norm);
}

int main(int argc, char* argv[])
{
    cudaError_t cudaStatus;
	size_t width = 32;
	size_t height= 512;
	float timeMS = 0;
	float timeTotalMS = 0;
	size_t memoryUsage = 0;
	ULONG_PTR gdiplusToken;
	GdiplusStartupInput gdiplusStartupInput;
	CLSID  encoderID;
	std::ofstream logData("C:\\Users\\Rollen\\Documents\\Report2B\\LogData.txt");

	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	getEncoderClsid(L"image/bmp", &encoderID);

	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

	for( int xs = 1; xs <= 1; xs++ )
	{
		for( int ys = 1; ys <= 1; ys++ )
		{	
			size_t length = width * height;
			float* texture = (float*)malloc(sizeof(float)*length);
			int stage = 3;
			cudaStatus = CreateFourierTextureFast(texture, width, height, &timeMS, &timeTotalMS, &memoryUsage,stage);
			if(cudaStatus != cudaSuccess) 
			{
				cudaDeviceReset();
				fprintf(stderr, "Failed to generate texture");
				return 1;
			} 
			else
			{
				printf("Time in Milliseconds: %g\n", timeMS);
				BYTE* bytes = (BYTE*)malloc(sizeof(BYTE)*length*4);

				float min = FLT_MAX;
				float max = FLT_MIN;
				for( int i = 0; i < (width*height); i++ )
				{
					if( texture[i] < min )
						min = texture[i];
					if( texture[i] > max )
						max = texture[i];
				}

				for( int i = 0; i < (width*height); i++ )
				{
					float v = ScaledNormalizeValue(texture[i], min, max) * 255.0f;
					BYTE val = (BYTE)v;

					bytes[4*i] = val;
					bytes[4*i+1] = val;
					bytes[4*i+2] = val;
				}
				
				free(texture);
				printf("Saving texture...\n");
				
				Bitmap* bmp =  new Bitmap(width, height, width*4, PixelFormat32bppRGB, bytes);
		
				std::stringstream ss;
				std::string fileNameStr;
				ss << "C:\\Users\\Rollen\\Documents\\Report2B\\Images\\FI_STEP" << stage << "_" << width << "_" << height << ".bmp";
				ss >> fileNameStr;
				WCHAR* fileName = (WCHAR*)malloc(sizeof(WCHAR)*(fileNameStr.length()+1));
				memset(fileName, 0, sizeof(WCHAR)*(fileNameStr.length()+1));
				mbstowcs(fileName, fileNameStr.c_str(), fileNameStr.length());
				bmp->Save(fileName,&encoderID);

				delete bmp;
				wprintf(L"Texture saved in %s\n", fileName);
	
				free(fileName);
				free(bytes);

				logData << width << "," << height << "," << timeMS << "," << ((double)memoryUsage)/1048576.0 << "," << timeTotalMS << std::endl;
			}

			width *= 2;
		}
		width = 512;
		height *= 2;
	}
	
	logData.close();

	GdiplusShutdown(gdiplusToken);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	
	system("PAUSE");
    return 0;
}

cudaError_t CreateFourierTextureFast(float* hostTexture, size_t texWidth, size_t texHeight, float* timeMSPerGeneration, float* totalTimeMS, size_t* usedBytes, int stepMax)
{
	cudaError_t cudaStatus = cudaSuccess;
	cudaEvent_t cudaStartEvent;
	cudaEvent_t cudaEndEvent;
	cudaEvent_t cudaOverallStartEvent;
	cudaEvent_t cudaOverallEndEvent;
	cuComplex* devDomain = NULL;
	cufftHandle fftPlanForward;
	cufftHandle fftPlanInverse;
	cufftResult cufftStatus;
	size_t freeMemory;
	size_t totalMemory;
	size_t oldUsedMemory;
	size_t texHeightDFT = texHeight / 2 + 1;
	dim3 largeGrid = CreateGrid(texWidth, texHeight);
	dim3 kernelGrid = CreateGrid(texWidth, texHeightDFT);
	
	cudaMemGetInfo(&freeMemory, &totalMemory);
	oldUsedMemory = totalMemory - freeMemory;

	cudaStatus = cudaMalloc(&devDomain, sizeof(cuComplex) * texWidth * texHeightDFT );
	if( cudaStatus != cudaSuccess )
	{
		printCudaError("Failed to allocate space for texture on the device.", cudaStatus);
		return cudaStatus;
	}
	
	cudaEventCreate(&cudaOverallStartEvent);
	cudaEventCreate(&cudaOverallEndEvent);
	
	cudaEventRecord(cudaOverallStartEvent);

	cufftStatus = cufftPlan2d(&fftPlanForward, texWidth, texHeight, CUFFT_R2C);
	cufftStatus = cufftSetCompatibilityMode(fftPlanForward, CUFFT_COMPATIBILITY_NATIVE);
	cufftStatus = cufftPlan2d(&fftPlanInverse, texWidth, texHeight, CUFFT_C2R);
	cufftStatus = cufftSetCompatibilityMode(fftPlanInverse, CUFFT_COMPATIBILITY_NATIVE);

	cudaEventCreate(&cudaStartEvent);
	cudaEventCreate(&cudaEndEvent);

	float totalTime = 0;
	float timeMSTemp = 0;
	int temp = 0;
	float min = FLT_MAX;
	float max = FLT_MIN;
	for( unsigned int i = 1; i <= 100; i++ )
	{
		srand(i);
		for( int x = 0; x < texWidth*texHeight; x++ ) 
		{
			temp = rand();
			hostTexture[x] = *(float*)&temp;
			if( hostTexture[x] < min )
				min = hostTexture[x];
			if( hostTexture[x] > max )
				max = hostTexture[x];
		}
		for( int x = 0; x < texHeight*texWidth; x++ )
			hostTexture[x] = ScaledNormalizeValue(hostTexture[x], min, max);

		cudaMemcpy(devDomain, hostTexture, sizeof(float) * texWidth * texHeight, cudaMemcpyHostToDevice);

		cudaEventRecord(cudaStartEvent);

		if(stepMax >= 1)
		{
			cufftExecR2C(fftPlanForward, (float*)devDomain, devDomain);
			if(stepMax >= 2)
			{
				ExecuteFilterSymmetric<<<kernelGrid, KERNEL_THREAD_COUNT>>>((cuComplex*)devDomain, texWidth, texHeightDFT, texHeight, 2.1);
				if(stepMax >= 3)
				{
					cufftExecC2R(fftPlanInverse, devDomain, (float*)devDomain);
				}
			}
		}
		cudaEventRecord(cudaEndEvent);
		cudaEventSynchronize(cudaEndEvent);

		cudaEventElapsedTime(&timeMSTemp, cudaStartEvent, cudaEndEvent);
		totalTime += timeMSTemp;
	}
	cudaEventDestroy(cudaStartEvent);
	cudaEventDestroy(cudaEndEvent);

	*timeMSPerGeneration = totalTime / 100;
	cudaMemGetInfo(&freeMemory, &totalMemory);
	*usedBytes = (totalMemory - freeMemory) - oldUsedMemory;

	cufftDestroy(fftPlanForward);
	cufftDestroy(fftPlanInverse);
	
	if(stepMax >= 3 || stepMax <= 1)
		cudaMemcpy(hostTexture, devDomain, sizeof(float) * texWidth * texHeight, cudaMemcpyDeviceToHost);
	else
	{
		float *devFloat = NULL;
		cudaMalloc(&devFloat, sizeof(float)*texWidth*texHeight);
		cudaMemset(devFloat, 0, sizeof(float)*texWidth*texHeight);
		GetMagnitudeSymmetric<<<kernelGrid, KERNEL_THREAD_COUNT>>>(devDomain, devFloat, texWidth, texHeightDFT, texHeight);
		cudaMemcpy(hostTexture, devFloat, sizeof(float) * texWidth * texHeight, cudaMemcpyDeviceToHost);
		cudaFree(devFloat);
	}

	cudaEventRecord(cudaOverallEndEvent);
	cudaEventSynchronize(cudaOverallEndEvent);
	cudaEventElapsedTime(&timeMSTemp, cudaOverallStartEvent, cudaOverallEndEvent);

	cudaEventDestroy(cudaOverallStartEvent);
	cudaEventDestroy(cudaOverallEndEvent);

	*totalTimeMS = timeMSTemp;

	cudaFree(devDomain);

	return cudaSuccess;
}

int getEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
   UINT  num = 0;          // number of image encoders
   UINT  size = 0;         // size of the image encoder array in bytes

   ImageCodecInfo* pImageCodecInfo = NULL;

   GetImageEncodersSize(&num, &size);
   if(size == 0)
      return -1;  // Failure

   pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
   if(pImageCodecInfo == NULL)
      return -1;  // Failure

   GetImageEncoders(num, size, pImageCodecInfo);

   for(UINT j = 0; j < num; ++j)
   {
      if( wcscmp(pImageCodecInfo[j].MimeType, format) == 0 )
      {
         *pClsid = pImageCodecInfo[j].Clsid;
         free(pImageCodecInfo);
         return j;  // Success
      }    
   }

   free(pImageCodecInfo);
   return -1;  // Failure
}

void printCudaError(const char * const message, cudaError_t error)
{
	printf("ERROR: ");
	printf(message);
	printf("\n");
	printf("\tCUDA Error: ");
	printf(cudaGetErrorString(error));
	printf("\n");
}

dim3 CreateGrid(size_t threadXCount, size_t threadYCount) 
{
	return dim3((threadXCount + KERNEL_THREAD_COUNT.x - 1)/KERNEL_THREAD_COUNT.x, 
				(threadYCount + KERNEL_THREAD_COUNT.y - 1)/KERNEL_THREAD_COUNT.y);
}

__global__ void ExecuteFilterSymmetric(cuComplex* freqDomain, size_t width, size_t height, size_t trueHeight, float roughness)
{
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if( x < width && y < height )
	{
		size_t s = x + y * width;
		size_t s2 = trueHeight * ( s / height ) + s % height;
		size_t trueX = s2 % width;
		size_t trueY = s2 / width;
		cuComplex* p = freqDomain + s;

		float avg = (width + trueHeight)/2.0f;
		float min1 = max(width,trueHeight);

		int fx = trueX;
		int fy = trueHeight / 2 - abs((int)(trueY - (trueHeight / 2)));
		float f = sqrtf(width*fx*fx/min1 + trueHeight*fy*fy/min1);

		// Apply scaling for all components except the DC Component
		if( fx != 0 || fy != 0 )
		{
			float factor = powf(f, roughness);
			p->x = p->x / factor;
			p->y = p->y / factor;
		}
	}
}

__global__ void GetMagnitudeSymmetric(cuComplex* input, float* output, size_t width, size_t height, size_t trueHeight)
{
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if( x < width && y < height )
	{
		size_t s = x + y * width;
		size_t s2 = trueHeight * ( s / height ) + s % height;
		output[s2] = sqrtf(input[s].x * input[s].x + input[s].y * input[s].y);
	}
}

__global__ void CreateRandom2DTexture(float* texture, size_t width, size_t height, unsigned long long randomSeed)
{
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if( x < width && y < height )
	{
		size_t s = x + y * width;
		float* p = texture + s;
		
		curandState_t randState;
		curand_init(randomSeed, s, 0, &randState);
		*p = curand_uniform(&randState);
	}
}
