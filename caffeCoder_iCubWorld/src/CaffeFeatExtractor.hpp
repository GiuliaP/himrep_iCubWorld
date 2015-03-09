/*
 * CaffeFeatExtractor.h
 */

#ifndef CAFFEFEATEXTRACTOR_H_
#define CAFFEFEATEXTRACTOR_H_

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

// Caffe and dependences
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "boost/make_shared.hpp"

#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <string>

// OpenCV includes
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)

/*using boost::shared_ptr;
using caffe::Blob;
using caffe::Caffe;
using caffe::Dataset;
using caffe::DatasetFactory;
using caffe::Datum;
using caffe::Net;*/

template<class Dtype>
class CaffeFeatExtractor {

	string pretrained_binary_proto_file;
	string feature_extraction_proto_file;

	shared_ptr<Net<Dtype> > feature_extraction_net;

	int mean_width;
	int mean_height;
	int mean_channels;

	vector<string> blob_names;

	string compute_mode;
	uint device_id;

public:

	CaffeFeatExtractor(string _pretrained_binary_proto_file,
			string _feature_extraction_proto_file,
			string _extract_feature_blob_names,
			string _compute_mode,
			uint _device_id,
			string module_name);

	virtual ~CaffeFeatExtractor();

	// the number of added images must be multiple of batch_size

	float extractBatch_multipleFeat(vector<cv::Mat> &images, int new_batch_size, vector< Blob<Dtype>* > &features);

	float extractBatch_singleFeat(vector<cv::Mat> &images, int new_batch_size, vector< Blob<Dtype>* > &features);

	float extractBatch_multipleFeat_1D(vector<cv::Mat> &images, int new_batch_size, vector< vector<Dtype> > &features);

	float extractBatch_singleFeat_1D(vector<cv::Mat> &images, int new_batch_size, vector< vector<Dtype> > &features);

	float extract_multipleFeat(cv::Mat &image, vector< Blob<Dtype>* > &features);

	float extract_singleFeat(cv::Mat &image, Blob<Dtype> *features);

	float extract_multipleFeat_1D(cv::Mat &image, vector< vector<Dtype> > &features);

	float extract_singleFeat_1D(cv::Mat &image, vector<Dtype> &features);
};

template <class Dtype>
CaffeFeatExtractor<Dtype>::CaffeFeatExtractor(string _pretrained_binary_proto_file,
		string _feature_extraction_proto_file,
		string _extract_feature_blob_names,
		string _compute_mode,
		uint _device_id,
		string _module_name) {

	  // Init Google Logging library with module name
	  //google::InitGoogleLogging(_module_name.c_str());

	  // Setup the GPU or the CPU mode for Caffe
	  if (strcmp(_compute_mode.c_str(), "GPU") == 0 || strcmp(_compute_mode.c_str(), "gpu") == 0) {

	    cout<< "Using GPU" << endl;

	    compute_mode = "GPU";
	    device_id = _device_id;

	    if (device_id<0)
	    {
	    	device_id=0;
	    	cout << "Attention! Specified device ID is negative!" << endl;
	    }

	    cout << "Using device_id = " << device_id << endl;

	    Caffe::set_mode(Caffe::GPU);
	    Caffe::SetDevice(device_id);

	    // Optional: to check that the GPU is working properly...
	    Caffe::DeviceQuery();

	  } else
	  {
	    cout << "Using CPU" << endl;
	    Caffe::set_mode(Caffe::CPU);
	  }

	  // Set phase (train or test)
	  //Caffe::set_phase(Caffe::TEST);

	  // Assign parameters

	  pretrained_binary_proto_file = _pretrained_binary_proto_file;
	  feature_extraction_proto_file = _feature_extraction_proto_file;

	  // Network creation using the given .prototxt for the structure
	  feature_extraction_net = boost::make_shared<Net<Dtype> > (feature_extraction_proto_file, caffe::TEST);

	  // Network initialization using the given .caffemodel for the weights
	  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto_file);

	  mean_width = 0;
	  mean_height = 0;
	  mean_channels = 0;

	  shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

	  TransformationParameter tp = memory_data_layer->layer_param().transform_param();

	  if (tp.has_mean_file())
      {
		  const string& mean_file = tp.mean_file();
		  cout << "Loading mean file from" << mean_file << endl;

		  BlobProto blob_proto;
		  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

		  Blob<Dtype> data_mean;
		  data_mean.FromProto(blob_proto);

		  mean_channels = data_mean.channels();
		  mean_width = data_mean.width();
		  mean_height = data_mean.height();

		  /*cout << mean_channels << endl;
		  cout << mean_height << endl;
		  cout << mean_width << endl;*/

	  }

	  // Check that requested blobs exist
	  boost::split(blob_names, _extract_feature_blob_names, boost::is_any_of(","));
	  for (size_t i = 0; i < blob_names.size(); i++) {
	    if (!feature_extraction_net->has_blob(blob_names[i]))
		{
			cout << "Unknown feature blob name " << blob_names[i] << " in the network " << feature_extraction_proto_file;
		}
	  }

}

template <class Dtype>
CaffeFeatExtractor<Dtype>::~CaffeFeatExtractor() {
	// TODO Auto-generated destructor stub
}


template<class Dtype>
float CaffeFeatExtractor<Dtype>::extractBatch_multipleFeat(vector<cv::Mat> &images, int new_batch_size, vector< Blob<Dtype>* > &features) {

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

	vector<int> labels(images.size(), 0);

	// Get pointer to data layer to set the input
    shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

	// Set batch size

	if (memory_data_layer->batch_size()!=new_batch_size)
	{
		if (images.size()%new_batch_size==0)
		{
			memory_data_layer->ChangeBatchSize(new_batch_size);
			cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
		}
		else
		{
			if (images.size()%memory_data_layer->batch_size()==0)
			{
				cout << "WARNING: image number is not multiple of requested batch size,leaving the old one" << endl;
				cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
			} else
			{
				cout << "WARNING: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
				memory_data_layer->ChangeBatchSize(1);
				cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
			}

		}

	} else
	{
		if (images.size()%memory_data_layer->batch_size()!=0)
		{
			cout << "WARNING: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
			memory_data_layer->ChangeBatchSize(1);
			cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
		}
	}

	int num_batches = images.size()/new_batch_size;

	// Set input

	// l'immagine deve avere la stessa dimensione della immagine media quando è passata a AddMatVector
	// altrimenti viene fatto un resize:
	// se l'immagine è ingrandita, ok
	// se l'immagine è sottocampionata, attenzione all'aliasing!!!!!!

	for (int i=0; i<images.size(); i++)
	{
		if (images[i].rows != mean_height || images[i].cols != mean_height)
		{
			if (images[i].rows > mean_height || images[i].cols > mean_height) {
				cout << "Aliasing risk!" << endl;
			}
			cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
		}
	}

	//boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0])->AddMatVector(images,labels);
	memory_data_layer->AddMatVector(images,labels);

	size_t num_features = blob_names.size();

	// Run network and retrieve features

	float loss = 0.0;
	// depending on your net's architecture, the blobs will hold accuracy and / or labels, etc...
	std::vector<Blob<Dtype>*> results;

	for (int b=0; b<num_batches; b++)
	{
		results = feature_extraction_net->ForwardPrefilled(&loss);

		for (int i = 0; i < num_features; ++i) {

			const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[i]);

			int batch_size = feature_blob->num();
			int channels = feature_blob->channels();
			int width = feature_blob->width();
			int height = feature_blob->height();

			features.push_back(new Blob<Dtype>(batch_size, channels, height, width));

			features.back()->CopyFrom(*feature_blob);
		}

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

    float msecPerImage = msecTotal/(float)images.size();

    // Compute and print the performance
    //cout << "Time per image = " << msecPerImage << " msec" << endl;

	////////////////////////////////////////////////////////////////////

    return msecPerImage;
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extractBatch_singleFeat(vector<cv::Mat> &images, int new_batch_size, vector< Blob<Dtype>* > &features) {

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

    vector<int> labels(images.size(), 0);

	// Get pointer to data layer to set the input
    shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

	// Set batch size

	if (memory_data_layer->batch_size()!=new_batch_size)
	{
		if (images.size()%new_batch_size==0)
		{
			memory_data_layer->ChangeBatchSize(new_batch_size);
			cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
		}
		else
		{
			if (images.size()%memory_data_layer->batch_size()==0)
			{
				cout << "WARNING: image number is not multiple of requested batch size,leaving the old one" << endl;
				cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
			} else
			{
				cout << "WARNING: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
				memory_data_layer->ChangeBatchSize(1);
				cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
			}

		}

	} else
	{
		if (images.size()%memory_data_layer->batch_size()!=0)
		{
			cout << "WARNING: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
			memory_data_layer->ChangeBatchSize(1);
			cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
		}
	}

	int num_batches = images.size()/new_batch_size;

	// Set input

	// l'immagine deve avere la stessa dimensione della immagine media quando è passata a AddMatVector
	// altrimenti viene fatto un resize:
	// se l'immagine è ingrandita, ok
	// se l'immagine è sottocampionata, attenzione all'aliasing!!!!!!

	for (int i=0; i<images.size(); i++)
	{
		if (images[i].rows != mean_height || images[i].cols != mean_height)
		{
			if (images[i].rows > mean_height || images[i].cols > mean_height) {
				cout << "Aliasing risk!" << endl;
			}
			cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
		}
	}

	//boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0])->AddMatVector(images,labels);
	memory_data_layer->AddMatVector(images,labels);

	size_t num_features = blob_names.size();
	if (num_features!=1)
	{
		cout<< "Error! The list of features to be extracted has not size one!" << endl;
		return -1;
	}

	// Run network and retrieve features

	float loss = 0.0;
	// depending on your net's architecture, the blobs will hold accuracy and / or labels, etc...
	std::vector<Blob<Dtype>*> results;

	for (int b=0; b<num_batches; b++)
	{
		results = feature_extraction_net->ForwardPrefilled(&loss);

		const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[0]);

		int batch_size = feature_blob->num();
		int channels = feature_blob->channels();
		int width = feature_blob->width();
		int height = feature_blob->height();

		features.push_back(new Blob<Dtype>(batch_size, channels, height, width));

		features.back()->CopyFrom(*feature_blob);

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

    float msecPerImage = msecTotal/(float)images.size();

    // Compute and print the performance
    //cout << "Time per image = " << msecPerImage << " msec" << endl;

	////////////////////////////////////////////////////////////////////

    return msecPerImage;
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extract_multipleFeat(cv::Mat &image, vector< Blob<Dtype>* > &features)
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

    int label = 0;

	// Get pointer to data layer to set the input
    shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

	// Set batch size to 1

	if (memory_data_layer->batch_size()!=1)
	{
		memory_data_layer->ChangeBatchSize(1);
		cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
	}

	// Set input

	// l'immagine deve avere la stessa dimensione della immagine media quando è passata a AddMatVector
	// altrimenti viene fatto un resize:
	// se l'immagine è ingrandita, ok
	// se l'immagine è sottocampionata, attenzione all'aliasing!!!!!!

	if (image.rows != mean_height || image.cols != mean_height)
	{
		if (image.rows > mean_height || image.cols > mean_height) {
			cout << "Aliasing risk!" << endl;
		}
		cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
	}

	//boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0])->AddMatVector(images,labels);
	memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

	size_t num_features = blob_names.size();

	// Run network and retrieve features

	float loss = 0.0;
	// depending on your net's architecture, the blobs will hold accuracy and / or labels, etc...
	std::vector<Blob<Dtype>*> results = feature_extraction_net->ForwardPrefilled(&loss);

	for (int f = 0; f < num_features; ++f) {

		const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[f]);

		int batch_size = feature_blob->num(); // should be 1
		if (batch_size!=1)
		{
			cout << "Error! Retrieved more than one feature, exiting..." << endl;
			return -1;
		}

		int channels = feature_blob->channels();
		int width = feature_blob->width();
		int height = feature_blob->height();

	    features.push_back(new Blob<Dtype>(1, channels, height, width));

	    features.back()->CopyFrom(*feature_blob);

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
    //cout << "Time per image = " << msecTotal << " msec" << endl;

	////////////////////////////////////////////////////////////////////

    return msecTotal;
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extract_singleFeat(cv::Mat &image, Blob<Dtype> *features) {

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

    int label = 0;

	// Get pointer to data layer to set the input
	shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size to 1

	if (memory_data_layer->batch_size()!=1)
	{
		memory_data_layer->ChangeBatchSize(1);
		cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
	}

	// Set input

	// l'immagine deve avere la stessa dimensione della immagine media quando è passata a AddMatVector
	// altrimenti viene fatto un resize:
	// se l'immagine è ingrandita, ok
	// se l'immagine è sottocampionata, attenzione all'aliasing!!!!!!

	if (image.rows != mean_height || image.cols != mean_height)
	{
		if (image.rows > mean_height || image.cols > mean_height) {
			cout << "Aliasing risk!" << endl;
		}
		cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
	}

	//boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0])->AddMatVector(images,labels);
	memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

	size_t num_features = blob_names.size();
	if(num_features!=1)
	{
		cout<< "Error! The list of features to be extracted has not size one!" << endl;
		return -1;
	}

	// Run network and retrieve features

	float loss = 0.0;
	// depending on your net's architecture, the blobs will hold accuracy and / or labels, etc...
	std::vector<Blob<Dtype>*> results = feature_extraction_net->ForwardPrefilled(&loss);

	const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[0]);

	int batch_size = feature_blob->num(); // should be 1
	if (batch_size!=1)
	{
		cout << "Error! Retrieved more than one feature, exiting..." << endl;
		return -1;
	}

	int channels = feature_blob->channels();
	int width = feature_blob->width();
	int height = feature_blob->height();

	if (features==NULL)
	{
		features = new Blob<Dtype>(1, channels, height, width);
	} else
	{
		features->Reshape(1, channels, height, width);
	}

	features->CopyFrom(*feature_blob);

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
    //cout << "Time per image = " << msecTotal << " msec" << endl;

	////////////////////////////////////////////////////////////////////

    return msecTotal;
}


template<class Dtype>
float CaffeFeatExtractor<Dtype>::extractBatch_multipleFeat_1D(vector<cv::Mat> &images, int new_batch_size, vector< vector<Dtype> > &features) {

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

    vector<int> labels(images.size(), 0);

	// Get pointer to data layer to set the input
    shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

	// Set batch size

	if (memory_data_layer->batch_size()!=new_batch_size)
	{
		if (images.size()%new_batch_size==0)
		{
			memory_data_layer->ChangeBatchSize(new_batch_size);
			cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
		}
		else
		{
			if (images.size()%memory_data_layer->batch_size()==0)
			{
				cout << "WARNING: image number is not multiple of requested batch size,leaving the old one" << endl;
				cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
			} else
			{
				cout << "WARNING: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
				memory_data_layer->ChangeBatchSize(1);
				cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
			}

		}

	} else
	{
		if (images.size()%memory_data_layer->batch_size()!=0)
		{
			cout << "WARNING: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
			memory_data_layer->ChangeBatchSize(1);
			cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
		}
	}

	int num_batches = images.size()/new_batch_size;

	// Set input

	// l'immagine deve avere la stessa dimensione della immagine media quando è passata a AddMatVector
	// altrimenti viene fatto un resize:
	// se l'immagine è ingrandita, ok
	// se l'immagine è sottocampionata, attenzione all'aliasing!!!!!!

	for (int i=0; i<images.size(); i++)
	{
		if (images[i].rows != mean_height || images[i].cols != mean_height)
		{
			if (images[i].rows > mean_height || images[i].cols > mean_height) {
				cout << "Aliasing risk!" << endl;
			}
			cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
		}
	}

	//boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0])->AddMatVector(images,labels);
	memory_data_layer->AddMatVector(images,labels);

	size_t num_features = blob_names.size();

	// Run network and retrieve features

	float loss = 0.0;
	// depending on your net's architecture, the blobs will hold accuracy and / or labels, etc...
	std::vector<Blob<Dtype>*> results;

	for (int b=0; b<num_batches; ++b)
	{
		results = feature_extraction_net->ForwardPrefilled(&loss);

		for (int f = 0; f < num_features; ++f) {

			const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[f]);

			int batch_size = feature_blob->num();

			int feat_dim = feature_blob->count() / batch_size; // should be equal to channels*width*height

			if (feat_dim!=feature_blob->channels()) // the feature is not 1D i.e. width!=1 or height!=1
			{
				cout<< "Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, width, height)" << endl;
			}

	        for (int i=0; i<batch_size; ++i)
	        {
	        	features.push_back(vector <Dtype>(feature_blob->mutable_cpu_data() + feature_blob->offset(i), feature_blob->mutable_cpu_data() + feature_blob->offset(i) + feat_dim));
	        }

		}

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

    float msecPerImage = msecTotal/(float)images.size();

    // Compute and print the performance
    //cout << "Time per image = " << msecPerImage << " msec" << endl;

	////////////////////////////////////////////////////////////////////

    return msecPerImage;
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extractBatch_singleFeat_1D(vector<cv::Mat> &images, int new_batch_size, vector< vector<Dtype> > &features) {

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

    vector<int> labels(images.size(), 0);

	// Get pointer to data layer to set the input
    shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

	// Set batch size

	if (memory_data_layer->batch_size()!=new_batch_size)
	{
		if (images.size()%new_batch_size==0)
		{
			memory_data_layer->ChangeBatchSize(new_batch_size);
			cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
		}
		else
		{
			if (images.size()%memory_data_layer->batch_size()==0)
			{
				cout << "WARNING: image number is not multiple of requested batch size,leaving the old one" << endl;
				cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
			} else
			{
				cout << "WARNING: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
				memory_data_layer->ChangeBatchSize(1);
				cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
			}

		}

	} else
	{
		if (images.size()%memory_data_layer->batch_size()!=0)
		{
			cout << "WARNING: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
			memory_data_layer->ChangeBatchSize(1);
			cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
		}
	}

	int num_batches = images.size()/new_batch_size;

	// Set input

	// l'immagine deve avere la stessa dimensione della immagine media quando è passata a AddMatVector
	// altrimenti viene fatto un resize:
	// se l'immagine è ingrandita, ok
	// se l'immagine è sottocampionata, attenzione all'aliasing!!!!!!

	for (int i=0; i<images.size(); i++)
	{
		if (images[i].rows != mean_height || images[i].cols != mean_height)
		{
			if (images[i].rows > mean_height || images[i].cols > mean_height) {
				cout << "Aliasing risk!" << endl;
			}
			cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
		}
	}

	//boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0])->AddMatVector(images,labels);
	memory_data_layer->AddMatVector(images,labels);

	size_t num_features = blob_names.size();
	if (num_features!=1)
	{
		cout<< "Error! The list of features to be extracted has not size one!" << endl;
		return -1;
	}

	// Run network and retrieve features

	float loss = 0.0;
	// depending on your net's architecture, the blobs will hold accuracy and / or labels, etc...
	std::vector<Blob<Dtype>*> results;

	for (int b=0; b<num_batches; ++b)
	{
		results = feature_extraction_net->ForwardPrefilled(&loss);

		const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[0]);

		int batch_size = feature_blob->num();

		int feat_dim = feature_blob->count() / batch_size; // should be equal to: channels*width*height
		if (feat_dim!=feature_blob->channels())
		{
			cout<< "Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, width, height)" << endl;
		}

	    for (int i=0; i<batch_size; ++i)
	    {
	    	features.push_back(vector <Dtype>(feature_blob->mutable_cpu_data() + feature_blob->offset(i), feature_blob->mutable_cpu_data() + feature_blob->offset(i) + feat_dim));
	    }

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

    float msecPerImage = msecTotal/(float)images.size();

    // Compute and print the performance
    //cout << "Time per image = " << msecPerImage << " msec" << endl;

	////////////////////////////////////////////////////////////////////

    return msecPerImage;
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extract_multipleFeat_1D(cv::Mat &image, vector< vector<Dtype> > &features)
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

    int label = 0;

	// Get pointer to data layer to set the input
    shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

	// Set batch size to 1

	if (memory_data_layer->batch_size()!=1)
	{
		memory_data_layer->ChangeBatchSize(1);
		cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
	}

	// Set input

	// l'immagine deve avere la stessa dimensione della immagine media quando è passata a AddMatVector
	// altrimenti viene fatto un resize:
	// se l'immagine è ingrandita, ok
	// se l'immagine è sottocampionata, attenzione all'aliasing!!!!!!

	if (image.rows != mean_height || image.cols != mean_height)
	{
		if (image.rows > mean_height || image.cols > mean_height) {
			cout << "Aliasing risk!" << endl;
		}
		cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
	}

	//boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0])->AddMatVector(images,labels);
	memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

	size_t num_features = blob_names.size();

	// Run network and retrieve features

	float loss = 0.0;
	// depending on your net's architecture, the blobs will hold accuracy and / or labels, etc...
	std::vector<Blob<Dtype>*> results = feature_extraction_net->ForwardPrefilled(&loss);

	for (int f = 0; f < num_features; ++f) {

		const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[f]);

		int batch_size = feature_blob->num(); // should be 1
		if (batch_size!=1)
		{
			cout << "Error! Retrieved more than one feature, exiting..." << endl;
			return -1;
		}

		int feat_dim = feature_blob->count(); // should be equal to: count/batch_size=channels*width*height
		if (feat_dim!=feature_blob->channels())
		{
			cout<< "Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, width, height)" << endl;
		}

		features.push_back(vector <Dtype>(feature_blob->mutable_cpu_data() + feature_blob->offset(0), feature_blob->mutable_cpu_data() + feature_blob->offset(0) + feat_dim));

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
    //cout << "Time per image = " << msecTotal << " msec" << endl;

	////////////////////////////////////////////////////////////////////

    return msecTotal;
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extract_singleFeat_1D(cv::Mat &image, vector<Dtype> &features) {

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

    int label = 0;

	// Get pointer to data layer to set the input
	shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size to 1

	if (memory_data_layer->batch_size()!=1)
	{
		memory_data_layer->set_batch_size(1);
		cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
	}

	// Set input

	// l'immagine deve avere la stessa dimensione della immagine media quando è passata a AddMatVector
	// altrimenti viene fatto un resize:
	// se l'immagine è ingrandita, ok
	// se l'immagine è sottocampionata, attenzione all'aliasing!!!!!!

	if (image.rows != mean_height || image.cols != mean_height)
	{
		if (image.rows > mean_height || image.cols > mean_height) {
			cout << "Aliasing risk!" << endl;
		}
		cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
	}

	//boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0])->AddMatVector(images,labels);
	memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

	size_t num_features = blob_names.size();
	if(num_features!=1)
	{
		cout<< "Error! The list of features to be extracted has not size one!" << endl;
		return -1;
	}

	// Run network and retrieve features

	Dtype loss = 0.0;
	// depending on your net's architecture, the blobs will hold accuracy and / or labels, etc...
	std::vector<Blob<Dtype>*> results = feature_extraction_net->ForwardPrefilled(&loss);

	const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[0]);

	int batch_size = feature_blob->num(); // should be 1
	if (batch_size!=1)
	{
		cout << "Error! Retrieved more than one feature, exiting..." << endl;
		return -1;
	}

	int feat_dim = feature_blob->count(); // should be equal to: count/num=channels*width*height
	if (feat_dim!=feature_blob->channels())
	{
		cout<< "Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, width, height)" << endl;
	}

	features.insert(features.end(), feature_blob->mutable_cpu_data() + feature_blob->offset(0), feature_blob->mutable_cpu_data() + feature_blob->offset(0) + feat_dim);

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
    //cout << "Time per image = " << msecTotal << " msec" << endl;

	////////////////////////////////////////////////////////////////////

    return msecTotal;
}

#endif /* CAFFEFEATEXTRACTOR_H_ */
