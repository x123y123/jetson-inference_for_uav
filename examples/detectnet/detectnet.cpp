/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "videoSource.h"
#include "videoOutput.h"
#include <stdio.h>
#include "detectNet.h"

#include <signal.h>
#include <time.h>
#include <sched.h>

#ifdef HEADLESS
	#define IS_HEADLESS() "headless"	// run without display
#else
	#define IS_HEADLESS() (const char*)NULL
#endif

#define DVFS
#define setaffinity
//#define rm_loop
//#define output_stream

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: detectnet [--help] [--network=NETWORK] [--threshold=THRESHOLD] ...\n");
	printf("                 input_URI [output_URI]\n\n");
	printf("Locate objects in a video/image stream using an object detection DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", detectNet::Usage());
	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
	printf("%s", Log::Usage());

	return 0;
}

int main( int argc, char** argv )
{
#ifdef setaffinity
    int cpu_id = 2;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif 
    int cfreq, gfreq, t = 0, cnt = 0;
    int gavailable_freq[15] = {114750000, 204000000, 306000000, 408000000, 510000000, 599250000, 701250000, 803250000, 854250000, 905250000, 956250000, 1007250000, 1058250000, 1109250000};

	FILE *fp = fopen("/home/uav/Desktop/write.txt","a+");	
	FILE *fp2 = fopen("/home/uav/Desktop/test.txt","r");	
	float lat , longit, sum_x = 0, sum_y = 0;		
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv, IS_HEADLESS());


    if( cmdLine.GetFlag("help") )
		return usage();


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("detectnet:  failed to create input stream\n");
		return 1;
	}

#ifdef output_stream
	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
		LogError("detectnet:  failed to create output stream\n");	
#endif    

	/*
	 * create detection network
	 */
	detectNet* net = detectNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("detectnet:  failed to load detectNet model\n");
		return 1;
	}

	// parse overlay flags
	const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));
    int frame = 0;                      //calculate number of frames          
    time_t endwait, test_sec;
    time_t stop_sec = 60;
    endwait = time(NULL) + stop_sec;
    test_sec = time(NULL) +1;
	/*
	 * processing loop
     */

    FILE *cfq = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq","r");


    FILE *cuda = fopen("cuda.txt","w");
    FILE *cuda_timelog = fopen("cuda_log.txt", "w");
#ifndef rm_loop
	//while( !signal_recieved )
    while(time(NULL) < endwait)
	{ 
#endif        
        printf("\nsched in cpu: %d\n", sched_getcpu());
        FILE *gfq = fopen("/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq","r");
        fprintf(cuda_timelog, "%ld ", time(NULL)); 
        fscanf(cfq, "%d", &cfreq);
        fscanf(gfq, "%d", &gfreq);
#ifdef DVFS
        char buf[50];
        FILE *cmd = popen("pidof mortor_control", "r");
        fgets(buf, 50, cmd);
        pid_t pid_signal = strtoul(buf, NULL, 10);
#endif
        // capture next image image
        uchar3* image = NULL;

        if( !input->Capture(&image, 1000) )
        {  
#ifndef rm_loop
            // check for EOS
            if( !input->IsStreaming() )
                break; 

            LogError("detectnet:  failed to capture video frame\n");
			continue;
#endif            
		}
        frame++;

		// detect objects in the frame
		detectNet::Detection* detections = NULL;
	
		const int numDetections = net->Detect(image, input->GetWidth(), input->GetHeight(), &detections, overlayFlags);
        float score = 0.0f;
		
		if( numDetections > 0 )
		{
			fscanf(fp2,"%f %f",&lat,&longit);				
			printf("%f %f\n",lat,longit);
			printf("================================================================================================================================================================================================================================================================================================================================\n");
			fprintf(fp ,"%f %f\n",lat,longit);					
			LogVerbose("%i objects detected\n", numDetections);
		
			for( int n=0; n < numDetections; n++ )
			{
				LogVerbose("detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				LogVerbose("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
                score = detections[n].Confidence;
			}
		}	
#ifdef output_stream
		// render outputs
		if( output != NULL )
		{
			output->Render(image, input->GetWidth(), input->GetHeight());

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			output->SetStatus(str);

			// check if the user quit
			if( !output->IsStreaming() )
				signal_recieved = true;
		}
        if (time(NULL) == test_sec) {
            fprintf(cuda, "%f %.0f %d %f\n", 1000/net->GetProfilerTime(PROFILER_TOTAL).y, net->GetNetworkFPS(), score, gfreq);
            test_sec = time(NULL) + 1;
        }
#endif
		// print out timing info
		net->PrintProfilerTimes();

        cnt++;
        sum_x += net->GetProfilerTime(PROFILER_TOTAL).x;
        sum_y += net->GetProfilerTime(PROFILER_TOTAL).y;
       

        fprintf(cuda_timelog, "%ld\n", time(NULL)); 
        fclose(gfq);
#ifdef DVFS        
        pclose(cmd);
        //if (net->GetProfilerTime(PROFILER_TOTAL).y < 17.0f) {     // for 60fps (720p)
        //if (net->GetProfilerTime(PROFILER_TOTAL).y < 33.0f) {     // for 30fps (1080p)
        if (net->GetProfilerTime(PROFILER_TOTAL).y < 10.0f) {       
            kill(pid_signal, SIGUSR1); 
            if (errno == EPERM)
               printf("\nSIGUSR1: permission denied!\n");
            else if (errno == ESRCH)
                printf("\nSIGUSR1: process does not exist!\n");
            else     
                printf("\nSIGUSR1: kill act!\n");
        }
        else {
            kill(pid_signal, SIGUSR2);

            if (errno == EPERM)
                printf("\nSIGUSR2: permission denied!\n");
            else if (errno == ESRCH)
                printf("\nSIGUSR2: process does not exist!\n");
            else 
                printf("\nSIGUSR2: kill act!\n");
        }
#endif        
#ifndef rm_loop    
    }
#endif
    fclose(cfq);
    fclose(cuda);
    fclose(cuda_timelog);
	/*
	 * destroy resources
	 */
	LogVerbose("detectnet:  shutting down...\n");
	
	SAFE_DELETE(input);
#ifdef output_stream	
    SAFE_DELETE(output);
#endif	
    SAFE_DELETE(net);

	LogVerbose("detectnet:  shutdown complete.\n");
	fclose(fp);
	fclose(fp2);	
    printf("package: %d\n", frame);
	return 0;
}

