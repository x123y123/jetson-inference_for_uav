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

#ifdef HEADLESS
	#define IS_HEADLESS() "headless"	// run without display
#else
	#define IS_HEADLESS() (const char*)NULL
#endif


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
    int cfreq, gfreq, t = 0, cnt = 0;
    int gavailable_freq[15] = {114750000, 204000000, 306000000, 408000000, 510000000, 599250000, 701250000, 803250000, 854250000, 905250000, 956250000, 1007250000, 1058250000, 1109250000};

	FILE *fp = fopen("/home/uav/Desktop/write.txt","a+");	
	FILE *fp2 = fopen("/home/uav/Desktop/test.txt","r");	
    FILE *txt = fopen("/home/uav/code/GPS/master_thesis/schedtil_in_60s/detectnet.txt","a+");
/*    FILE *cfq = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq","r");
    FILE *gfq = fopen("/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq","r");
    fscanf(cfq, "%d", &cfreq);
    fscanf(gfq, "%d", &gfreq);*/
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


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
		LogError("detectnet:  failed to create output stream\n");	
	

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

    time_t endwait, test_sec;
    time_t stop_sec = 60;
    endwait = time(NULL) + stop_sec;
    test_sec = time(NULL) +1;
	/*
	 * processing loop
	 */
	//while( !signal_recieved )
    while(time(NULL) < endwait)
	{ 

        char buf[10];
        FILE *cfq = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq","r");
        FILE *gfq = fopen("/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq","r");
        FILE *cmd = popen("pidof mortor_control", "r");
        fscanf(cfq, "%d", &cfreq);
        fscanf(gfq, "%d", &gfreq);
        fgets(buf, 10, cmd);
        pid_t pid_signal = strtoul(buf, NULL, 10);
        pclose(cmd);

		// capture next image image
		uchar3* image = NULL;

		if( !input->Capture(&image, 1000) )
		{
			// check for EOS
			if( !input->IsStreaming() )
				break; 

			LogError("detectnet:  failed to capture video frame\n");
			continue;
		}

		// detect objects in the frame
		detectNet::Detection* detections = NULL;
	
		const int numDetections = net->Detect(image, input->GetWidth(), input->GetHeight(), &detections, overlayFlags);
		
		if( numDetections > 0 )
		{
			fscanf(fp2,"%f %f",&lat,&longit);				
			printf("%f %f\n",lat,longit);
			printf("=========================================================\n");
			fprintf(fp ,"%f %f\n",lat,longit);					
			LogVerbose("%i objects detected\n", numDetections);
		
			for( int n=0; n < numDetections; n++ )
			{
				LogVerbose("detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				LogVerbose("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
			}
		}	

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
		// print out timing info
		net->PrintProfilerTimes();

        if (net->GetProfilerTime(PROFILER_TOTAL).y < 34)
            kill(pid_signal, SIGUSR2); 
        cnt++;
        sum_x += net->GetProfilerTime(PROFILER_TOTAL).x;
        sum_y += net->GetProfilerTime(PROFILER_TOTAL).y;
        kill(pid_signal, SIGUSR1);

        if (errno == EPERM)
            printf("permission denied!\n");
        else if (errno == ESRCH)
            printf("process does not exist!\n");
        else 
            printf("kill act!\n");
        
        fclose(cfq);
        fclose(gfq);
    }
	
    fprintf(txt,"%f %f %d %d\n", sum_x / cnt, sum_y / cnt, cfreq, gfreq);
          
    fclose(txt);
//        fclose(cfq);
//        fclose(gfq);
	/*
	 * destroy resources
	 */
	LogVerbose("detectnet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);

	LogVerbose("detectnet:  shutdown complete.\n");
	fclose(fp);
	fclose(fp2);	
	return 0;
}

