#include <iostream>
#include <string>
#include <yarp/sig/all.h>
#include <yarp/math/Math.h>
#include <yarp/os/all.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Network.h>
#include <yarp/os/Thread.h>
#include <yarp/os/Time.h>
#include <iCub/ctrl/math.h>
#include <yarp/os/Os.h>


#include <yarp/os/Semaphore.h>

#ifdef _WIN32
	#include "win_dirent.h"
#else
	#include "dirent.h"
#endif


#ifdef LIBSVMLIN_AVAILABLE
    #include "SVMLinear.h"
#endif

#ifdef GURLS_AVAILABLE
    #include "RLSLinear.h"
#endif


#define STATE_DONOTHING         0
#define STATE_SAVING            1
#define STATE_RECOGNIZING       2

#define SVM     0
#define RLS     1
#define BOTH    2

using namespace std; 
using namespace yarp::os; 
using namespace yarp::sig;
using namespace yarp::math;


class linearClassifierThread : public Thread
{
private:

        string inputFeatures;
        string outputPortName;
        string outputScorePortName;

        string currPath;
        string pathObj;
        //Port *commandPort;
        BufferedPort<Bottle> featuresPort;
        BufferedPort<Bottle> outputPort;
        Port scorePort;
        //Semaphore *mutex;
        Semaphore mutex;
        fstream objFeatures;

        //Dataset variables
        vector<pair<string,vector<string> > > knownObjects;
        vector<vector<vector<double> > > Features;
        vector<int> datasetSizes;


        //Learning variables
        int bufferSize;
        int use_only_prediction_mode;
        int prediction_mode;
        int useWeightedClassification;


        //SVM variables

#ifdef LIBSVMLIN_AVAILABLE
        vector<SVMLinear> linearClassifiers_SVM;
        vector<vector<double > > bufferScores_SVM;
        vector<vector<int > > countBuffer_SVM;
#endif

        //GURLS variables
#ifdef GURLS_AVAILABLE
        vector<RLSLinear> linearClassifiers_RLS;
        vector<vector<double > > bufferScores_RLS;
        vector<vector<int > > countBuffer_RLS;
#endif

        double CSVM;
        bool paramsel;

        int currentState;

        double print_times;

        string true_class;

        //Private Methods
        void checkKnownObjects();
        int getdir(string dir, vector<string> &files);
        void readFeatures(string filePath, vector<vector<double> > *featuresMat);
public:

    //linearClassifierThread(yarp::os::ResourceFinder &rf,Port* commPort);
    linearClassifierThread(yarp::os::ResourceFinder &rf);
    void prepareObjPath(string objName);
    bool threadInit();
    void threadRelease();
    void run(); 
    void interrupt();
    void createFullPath(const char * path);
    void stopAll();
    bool loadFeatures();
    bool trainClassifiers();
    bool startRecognition();
    bool forgetClass(string className, bool retrain=true);
    bool forgetAll();
    bool getClassList(Bottle &b);
    bool setMode(string mode_name);
    bool set_true_class(string _true_class);
    bool get_true_class(string &_true_class);
};
