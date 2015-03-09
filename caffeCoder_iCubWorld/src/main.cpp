/* 
 * Copyright (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
 * Author: Giulia Pasquale
 * email:  giulia.pasquale@iit.it
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * http://www.robotcub.org/icub/license/gpl.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
*/

/** 
\defgroup icub_caffeCoder caffeCoder
 
The module extracts a feature vector from the input image using the Caffe framework and a pretrained convolutional net.
 
\section intro_sec Description 
This module is responsible for coding images into feature vectors.
 
The commands sent as bottles to the module port 
/caffeCoder/rpc are the following: 
 
(notation: [.] identifies a vocab, <.> specifies a double,
"." specifies a string) 
 
<b>HELP</b> 
format: [help]
action: it prints the available commands.
 
 
<b>DUMP_CODE</b>
format: [dump] "a"
action: starts to save features in a binary file. "a" is optional and stands for "append".

\section lib_sec Libraries 
- YARP libraries. 

- OpenCV 2.2 libraries.

- Caffe libraries

\section portsc_sec Ports Created 
- \e /caffeCoder/img:i receives an image.
 
- \e /caffeCoder/img:o streams out the image.
 
- \e /caffeCoder/code:o streams out the vector containing the hierarchical image representation.
 
- \e /caffeCoder/rpc receives requests.
  
\section parameters_sec Parameters 
        
--dictionary_file \e file
- specify the dictionary file.ini where atoms are stored.
 
--coding_mode \e coding
- specify the coding method used. Possible values are "BOW" for Bag of Words, 
  "SC" for Sparse Coding, "BCE" for Best Codes Entries.
 
--nearest_neighbor \e knn
- specify the number of nearest neighbors used for the encoding with BCE.
 
--dense_descriptors \e [1|0] 
- specify is a regular grid of dense descriptors must be used (1) or sparse keypoints must be detected (0).
 
--grid_scale \e scale
- specify the scale of local descriptors extracted from a dense grid.
 
--grid_step \e step
- specify the sampling step among descriptors in a dense grid.

\section tested_os_sec Tested OS
Linux

\author Giulia Pasquale
*/ 

// YARP
#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Time.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Semaphore.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/PortReport.h>
#include <yarp/os/Stamp.h>

#include <yarp/sig/Vector.h>
#include <yarp/sig/Image.h>

#include <yarp/math/Math.h>
#include <yarp/math/Rand.h>

// OpenCV
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

// iCub
#include <iCub/boostMIL/ClassifierFactory.h>
#include <iCub/boostMIL/ClassifierInput.h>
#include <iCub/boostMIL/WeakClassifier.h>
#include <iCub/boostMIL/MILClassifier.h>
#include <iCub/boostMIL/OnlineBoost.h>

// General includes
#include <stdio.h>
#include <stdlib.h> // getenv
#include <string>
#include <deque>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

// CUDA
#include <cuda_runtime.h>
#include <cuda.h>

////////////////////////////////////////////7

#include "CaffeFeatExtractor.hpp"

using namespace std;
using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;
using namespace iCub::boostMIL;

#define CMD_HELP                    VOCAB4('h','e','l','p')
#define DUMP_CODE                   VOCAB4('d','u','m','p')
#define DUMP_STOP                   VOCAB4('s','t','o','p')

#define CAFFE_ROOT_ENV				"Caffe_ROOT"

class CaffeCoderPort: public BufferedPort<Image>
{
private:

    // Resource Finder and module options

	ResourceFinder                      &rf;

    string                              contextPath;

    bool                                verbose;

    bool                                dump_code;

    double                              rate;
    double                              last_read;

    // Data (common to all methods)

    cv::Mat				  			    matImg;

    Port                				port_out_img;
    Port                				port_out_code;

    FILE                                *fout_code;

    Semaphore                           mutex;

    // Data (specific for each method - instantiate only those are needed)

    CaffeFeatExtractor<double>                  *caffe_extractor;

    void onRead(Image &img)
    {

        // Read at specified rate
        if (Time::now() - last_read < rate)
            return;

        mutex.wait();

        // If something arrived...
        if(img.width()>0 && img.height()>0)
        {

        	// TIMING code
        	////////////////////////////////////////////////////////////////////

        	cudaEvent_t start, stop;

            //checkCudaErrors(cudaEventCreate(&start));
            //checkCudaErrors(cudaEventCreate(&stop));

            //checkCudaErrors(cudaEventRecord(start, NULL));

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, NULL);

            ////////////////////////////////////////////////////////////////////

        	// Convert the image
        	cv::cvtColor(cv::Mat((IplImage*)img.getIplImage()), matImg, CV_RGB2BGR);

            // Extract the feature vector
            std::vector<double> codingVec;
            float msecPerImage = caffe_extractor->extract_singleFeat_1D(matImg, codingVec);

            // Dump if required
            if(dump_code)
            {
            	fwrite (&codingVec[0], sizeof(double), codingVec.size(), fout_code);
            }

        	// TIMING code
        	////////////////////////////////////////////////////////////////////

            // Record the stop event
            //checkCudaErrors(cudaEventRecord(stop, NULL));
            cudaEventRecord(stop, NULL);

            // Wait for the stop event to complete
            //checkCudaErrors(cudaEventSynchronize(stop));
            cudaEventSynchronize(stop);

            float msecTotal = 0.0f;
            //checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
            cudaEventElapsedTime(&msecTotal, start, stop);

            // Compute and print the performance
            cout << msecPerImage << " msec \t" << msecTotal << " msec" <<endl;

        	////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

            Stamp stamp;
            this->getEnvelope(stamp);

            if(port_out_code.getOutputCount())
            {
            	port_out_code.setEnvelope(stamp);
                Vector codingYarpVec(codingVec.size(), &codingVec[0]);
                port_out_code.write(codingYarpVec);
            }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

            if(port_out_img.getOutputCount())
            {
            	port_out_img.write(img);
            }
        }

        mutex.post();
    }

public:
   CaffeCoderPort(ResourceFinder &_rf) :BufferedPort<Image>(),rf(_rf)
   {

	   // Resource Finder and module options

       contextPath = rf.getHomeContextPath().c_str();

       verbose = rf.check("verbose");

   	   dump_code = rf.check("dump_code");

       rate = rf.check("rate",Value(0.0)).asDouble();
       last_read = 0.0;

       // Data (common to all methods)

       string name = rf.find("name").asString().c_str();

       port_out_img.open(("/"+name+"/img:o").c_str());
       port_out_code.open(("/"+name+"/code:o").c_str());

       BufferedPort<Image>::useCallback();

       if(dump_code)
       {
           string code_path = rf.check("dump_code",Value("codes.txt")).asString().c_str();
           code_path = contextPath + "/" + code_path;
           string code_write_mode = rf.check("append")?"wb+":"wb";

           fout_code = fopen(code_path.c_str(),code_write_mode.c_str());
       }


       // Data (specific for each method - instiate only those are needed)

       // Caffe environment variable
       string caffe_ROOT = string( getenv(CAFFE_ROOT_ENV) );

       // Binary file (.caffemodel) containing the pretrained network's weights
       string pretrained_binary_proto_file;
       if (rf.find("pretrained_binary_proto_file").isNull())
       {
    	   pretrained_binary_proto_file = "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    	   if (caffe_ROOT!="")
    	   {
    		   pretrained_binary_proto_file = caffe_ROOT + "/" + pretrained_binary_proto_file;
    		   cout << "Using pretrained network: " << pretrained_binary_proto_file << endl;
    	   } else
    	   {
    		   cout << "Empty CAFFE_ROOT_ENV and missing 'pretrained_binary_proto_file' in .ini, aborting..." << endl;
    	   }
       } else
       {
    	   pretrained_binary_proto_file = rf.find("pretrained_binary_proto_file").asString().c_str();
    	   cout << "Using pretrained network: " << pretrained_binary_proto_file << endl;
       }

       // Text file (.prototxt) defining the network structure
       string feature_extraction_proto_file = rf.check("feature_extraction_proto_file", Value("imagenet_val.prototxt")).asString().c_str();
       feature_extraction_proto_file = rf.findFile(feature_extraction_proto_file);
       if (feature_extraction_proto_file == "")
       {
    	   feature_extraction_proto_file = contextPath + "/" + feature_extraction_proto_file;
       }
       cout << "Using network defined in: " << feature_extraction_proto_file << endl;

       string extract_features_blob_names = rf.check("extract_features_blob_names", Value("fc7")).asString().c_str();

       string compute_mode = rf.check("compute_mode", Value("GPU")).asString();
       int device_id = rf.check("device_id", Value(0)).asInt();

       caffe_extractor = NULL;
       caffe_extractor = new CaffeFeatExtractor<double>(pretrained_binary_proto_file,
    		   feature_extraction_proto_file,
    		   extract_features_blob_names,
    		   compute_mode,
    		   device_id,
    		   name);

   }

   void interrupt()
   {
        mutex.wait();

        port_out_code.interrupt();
        port_out_img.interrupt();
        BufferedPort<Image>::interrupt();
        mutex.post();
   }

   void resume()
   {
        mutex.wait();

        port_out_code.resume();
        port_out_img.resume();
        BufferedPort<Image>::resume();
        mutex.post();
   }

   void close()
   {
        mutex.wait();

        //if(ipl!=NULL)
        //    cvReleaseImage(&ipl);

        if(dump_code)
            fclose(fout_code);

		port_out_code.close();
        port_out_img.close();

        BufferedPort<Image>::close();

        mutex.post();
   }

   bool execReq(const Bottle &command, Bottle &reply)
   {
       switch(command.get(0).asVocab())
       {
           case(CMD_HELP):
           {
                reply.clear();
                reply.add(Value::makeVocab("many"));
                reply.addString("[dump] [path-to-file] [a] to start dumping the codes. Use 'a' for appending.");
                reply.addString("[stop] to stop dumping.. ");
                return true;
           }

           case(DUMP_CODE):
           {
                mutex.wait();

                dump_code = true;
                string code_path;
                string code_write_mode;

                if (command.size()==1)
                {
                	code_path = contextPath + "/codes.bin";
                	code_write_mode="wb";
                }
                else if (command.size()==2)
                {
                	if (strcmp(command.get(1).asString().c_str(),"a")==0)
                	{
                		code_write_mode="wb+";
                	    code_path = contextPath + "/codes.bin";
                	} else
                	{
                		code_write_mode="wb";
                		code_path = command.get(1).asString().c_str();
                	}
                } else if (command.size()==3)
                {
                	code_write_mode="wb+";
                	code_path = command.get(2).asString().c_str();
                }

                fout_code = fopen(code_path.c_str(),code_write_mode.c_str());
                reply.addString("Start dumping codes...");

                mutex.post();
                return true;
           }

           case(DUMP_STOP):
           {
                mutex.wait();

                dump_code = false;
                fclose(fout_code);
                reply.addString("Stopped SIFT Dump.");

                mutex.post();
                return true;
           }

           default:
               return false;
       }
   }

};


class CaffeCoderModule: public RFModule
{
protected:
    CaffeCoderPort         *caffePort;
    Port                    rpcPort;

public:
    CaffeCoderModule()
    {
        caffePort=NULL;
    }

    bool configure(ResourceFinder &rf)
    {

        Time::turboBoost();

        caffePort = new CaffeCoderPort(rf);

        string name = rf.find("name").asString().c_str();

        caffePort->open(("/"+name+"/img:i").c_str());

        rpcPort.open(("/"+name+"/rpc").c_str());
        attach(rpcPort);

        return true;
    }

    bool interruptModule()
    {
        if(caffePort!=NULL)
            caffePort->interrupt();

        rpcPort.interrupt();

        return true;
    }

    bool close()
    {
        if(caffePort!=NULL)
        {
            caffePort->close();
            delete caffePort;
        }

        rpcPort.close();

        return true;
    }

    bool respond(const Bottle &command, Bottle &reply)
    {
        if(caffePort->execReq(command,reply))
            return true;
        else
            return RFModule::respond(command,reply);
    }

    double getPeriod()    { return 1.0;  }

    bool   updateModule()
    {
        //caffePort->update();

        return true;
    }

};


int main(int argc, char *argv[])
{
   Network yarp;

   if (!yarp.checkNetwork())
       return -1;

   ResourceFinder rf;
   rf.setVerbose(true);
   rf.setDefaultContext("himrep_icubworld");
   rf.setDefaultConfigFile("caffeCoder.ini");
   rf.configure(argc,argv);
   rf.setDefault("name","caffeCoder");

   CaffeCoderModule mod;

   int deviceCount = 0;
   cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

   return mod.runModule(rf);

   //return feature_extraction_pipeline<float>(argc, argv);

}

