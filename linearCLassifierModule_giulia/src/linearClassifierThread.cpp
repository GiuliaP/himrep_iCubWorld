#include "linearClassifierThread.h"

#ifdef GURLS_AVAILABLE
    #include "gurls++/gmat2d.h"
#endif

//linearClassifierThread::linearClassifierThread(yarp::os::ResourceFinder &rf, Port* commPort)
linearClassifierThread::linearClassifierThread(yarp::os::ResourceFinder &rf)
{

    //this->commandPort=commPort;
    currentState=STATE_DONOTHING;

    this->currPath = rf.getHomeContextPath().c_str();
    
    //mutex=new Semaphore(1);

    string moduleName = rf.check("name",Value("linearClassifier"), "module name (string)").asString().c_str();
    this->inputFeatures = "/";
    this->inputFeatures += moduleName;
    this->inputFeatures += rf.check("InputPortFeatures",Value("/features:i"),"Input image port (string)").asString().c_str();

    this->outputPortName = "/";
    this->outputPortName += moduleName;
    this->outputPortName += rf.check("OutputPortClassification",Value("/classification:o"),"Input image port (string)").asString().c_str();

    this->outputScorePortName = "/";
    this->outputScorePortName += moduleName;
    this->outputScorePortName += rf.check("OutputPortScores",Value("/scores:o"),"Input image port (string)").asString().c_str();

    this->bufferSize = rf.check("BufferSize",Value(15),"Buffer Size").asInt();
    this->CSVM = rf.check("CSVM",Value(1.0),"CSVM").asDouble();
    this->paramsel = rf.check("paramsel");
    this->useWeightedClassification= rf.check("Weighted",Value(0)).asInt();

    this->print_times = rf.check("print_times");

    printf("WeightedClassification: %d \n",useWeightedClassification);

    string dbfolder = rf.check("databaseFolder",Value("database"), "module name (string)").asString().c_str();
    dbfolder="/"+dbfolder;
    this->currPath=this->currPath+dbfolder;



#if defined(GURLS_AVAILABLE) && defined(LIBSVMLIN_AVAILABLE)

    string input_use_only_prediction_mode = rf.check("use_only_mode",Value("RLS"), "use only prediction mode (string)").asString().c_str();
    
    if(!strcmp(input_use_only_prediction_mode.c_str(),"BOTH") || !strcmp(input_use_only_prediction_mode.c_str(),"both"))
        this->use_only_prediction_mode=BOTH;
    else if(!strcmp(input_use_only_prediction_mode.c_str(),"SVM") || !strcmp(input_use_only_prediction_mode.c_str(),"svm"))
        this->use_only_prediction_mode=SVM;
    else
        this->use_only_prediction_mode=RLS;

#elif defined(GURLS_AVAILABLE)
    this->use_only_prediction_mode=RLS;
#else
    this->use_only_prediction_mode=SVM;
#endif

    string input_prediction_mode = rf.check("mode",Value("RLS"), "prediction mode (string)").asString().c_str();
    this->setMode(input_prediction_mode);
}

bool linearClassifierThread::threadInit() 
{
     if (!featuresPort.open(inputFeatures.c_str())) {
        cout  << ": unable to open port " << inputFeatures << endl;
        return false; 
    }

    if (!outputPort.open(outputPortName.c_str())) {
        cout  << ": unable to open port " << outputPortName << endl;
        return false; 
    }

    if (!scorePort.open(outputScorePortName.c_str())) {
        cout  << ": unable to open port " << outputScorePortName << endl;
        return false; 
    }

    true_class = "?";

    trainClassifiers();

    return true;
}

bool linearClassifierThread::trainClassifiers()
{
    stopAll();

    //mutex->wait();
    mutex.wait();

#ifdef LIBSVMLIN_AVAILABLE
    
    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
        if(linearClassifiers_SVM.size()>0)
            for (int i=0; i<linearClassifiers_SVM.size(); i++)
                linearClassifiers_SVM[i].freeModel();

#endif 

#ifdef GURLS_AVAILABLE
    
    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
        if(linearClassifiers_RLS.size()>0)
            for (int i=0; i<linearClassifiers_RLS.size(); i++)
                linearClassifiers_RLS[i].freeModel();

#endif

    cout << "load features" << endl;
    
    loadFeatures();
    if(this->datasetSizes.size()==0)
    {
        //mutex->post();
        mutex.post();
        return false;
    }

    cout << "features loaded" << endl;

    double train_time_RLS;
    double train_time_SVM;

    // RLS ****************************************************************************

#ifdef GURLS_AVAILABLE

    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
    {
        double init_time = Time::now();

        linearClassifiers_RLS.clear();

        int T = knownObjects.size();

        int d = 0;
        if(Features.size()>0 && Features[0].size()>0)
            d = Features[0][0].size();
        else
        {
            cout << "Error! No samples provided" << endl;
            return false;
        }

        int n = 0;
        for(int i=0; i<knownObjects.size(); i++)
            n += Features[i].size();

        gurls::gMat2D<float> Y(n,1);
        gurls::gMat2D<float> X(n,d+1);

        for (int idx_curr_obj=0; idx_curr_obj<knownObjects.size(); idx_curr_obj++)
        {    
            string name=knownObjects[idx_curr_obj].first;
            RLSLinear rlsmodel(name);

        
            float pos_weight = useWeightedClassification?sqrt((float)(((float)n-Features[idx_curr_obj].size())/Features[idx_curr_obj].size())):1.0;
            float neg_weight = -1.0;

            //fill positive labels for Y
            int curr_row_idx = 0;
            for(int idx_obj=0; idx_obj<knownObjects.size(); idx_obj++)
            {
                for(int idx_feat=0; idx_feat<Features[idx_obj].size(); idx_feat++)
                {
                    float weight=(idx_obj==idx_curr_obj)?pos_weight:neg_weight;

                    //gurls::gVec<double> tmp_row(&(Feature[idx_obj][idx_feat]),Feature[idx_obj][idx_feat].size());
                    //X.setRow(weight*tmp_row,curr_row_idx);

                    for(int i=0; i<d; i++)
                        X(curr_row_idx,i)=abs(weight)*Features[idx_obj][idx_feat][i];
                    X(curr_row_idx,d)=abs(weight);

                    Y(curr_row_idx,0)=weight;

                    curr_row_idx++;
                }
            }

            printf("[RLS] nClass: %d nPositive: %d nNegative: %d\n",knownObjects.size(),Features[idx_curr_obj].size(),n-Features[idx_curr_obj].size());

            rlsmodel.trainModel(X,Y);
            linearClassifiers_RLS.push_back(rlsmodel);
            string tmpModelPath=currPath+"/"+knownObjects[idx_curr_obj].first+"/rlsmodel";
        }    

        train_time_RLS=Time::now()-init_time;
    }
    
#endif


    // SVM ****************************************************************************

#ifdef LIBSVMLIN_AVAILABLE

    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
    {
        double init_time = Time::now();

        linearClassifiers_SVM.clear();

        int nClass=2; // One vs All

        for (int i=0; i<knownObjects.size(); i++)
        {
            int nPositive=0;
            int nNegative=0;
            string name=knownObjects[i].first;
            SVMLinear svmmodel(name);

            vector<vector<double> > orderedF;
            vector<double> orderedLabels;
            for (int k=0; k<knownObjects.size(); k++)
            {
                for(int j=0; j<Features[k].size(); j++)
                    if(knownObjects[i].first==knownObjects[k].first)
                    {
                        orderedF.push_back(Features[k][j]);
                        orderedLabels.push_back(1.0);
                        nPositive++;
                    }
            }

            for (int k=0; k<knownObjects.size(); k++)
            {
                for(int j=0; j<Features[k].size(); j++)
                    if(knownObjects[i].first!=knownObjects[k].first)
                    {
                        orderedF.push_back(Features[k][j]);
                        orderedLabels.push_back(-1.0);
                        nNegative++;
                    }
            }

            printf("nClass: %d nPositive: %d nNegative: %d\n",nClass,nPositive,nNegative);
            parameter param;
            if(useWeightedClassification)
                param=svmmodel.initialiseParam(L2R_L2LOSS_SVC,CSVM,0.0,nClass,nPositive,nNegative);
            else
                param=svmmodel.initialiseParam(L2R_L2LOSS_SVC_DUAL,CSVM);

            svmmodel.trainModel(orderedF,orderedLabels,param);

            linearClassifiers_SVM.push_back(svmmodel);
            /*for (int k=0; k<Features[0][0].size(); k++)
                cout << svmmodel.modelLinearSVM->w[k] << " ";
            cout << endl;*/

            string tmpModelPath=currPath+"/"+knownObjects[i].first+"/svmmodel";
            //svmmodel.saveModel(tmpModelPath);
        }

        train_time_SVM=Time::now()-init_time;
    }

#endif


    //**************************************
    if(print_times)
    {
        if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
            printf("RLS training time: %f\n",train_time_RLS);

        if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
            printf("SVM training time: %f\n",train_time_SVM);
    }
    
    cout << "trained" << endl;
    //mutex->post();
    mutex.post();

    return true;

}

bool linearClassifierThread::loadFeatures()
{
    checkKnownObjects();
    if(this->knownObjects.size()==0)
        return false;

    Features.clear();
    Features.resize(knownObjects.size());
    datasetSizes.clear();

//     if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
//         SVMLinear svmmodel(knownObjects[0].first);

// #ifdef GURLS_AVAILABLE
//     if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
//         RLSLinear rlsmodel(knownObjects[0].first);
// #endif

    for (int i=0; i<knownObjects.size(); i++)
    {
        vector<string> obj=knownObjects[i].second;
        int cnt=0;
        for (int k=0; k< obj.size(); k++)
        {
            vector<vector<double> > tmpF;
            readFeatures(obj[k],&tmpF);

            cnt=cnt+tmpF.size();
            for (int t =0; t<tmpF.size(); t++)
                Features[i].push_back(tmpF[t]);
        }

        if(cnt>0)
            this->datasetSizes.push_back(cnt);
    }
    
    return true;

}

void linearClassifierThread::readFeatures(string filePath, vector<vector<double> > *featuresMat)
{
    string line;
    ifstream infile;
    infile.open (filePath.c_str());

    while(!infile.eof() && infile.is_open()) // To get you all the lines.
    {
        vector<double> f;
        getline(infile,line); // Saves the line in STRING.

        char * val= strtok((char*) line.c_str()," ");

        while(val!=NULL)
        {
        
            double value=atof(val);
            f.push_back(value);
            val=strtok(NULL," ");
        }
        if(f.size()>0)
            featuresMat->push_back(f);
    }

    infile.close();
}

bool linearClassifierThread::forgetClass(string className, bool retrain)
{

    string classPath=currPath+"/"+className;
    if(yarp::os::stat(classPath.c_str()))
    {
        return true;
    }

    vector<string> files;
    getdir(classPath,files);

    for (int i=0; i< files.size(); i++)
    {
        if(!files[i].compare(".") || !files[i].compare(".."))
            continue;
        string feature=classPath+"/"+files[i];
        remove(feature.c_str());
    }
    bool res=yarp::os::rmdir(classPath.c_str())==0;

    if(res && retrain)
        trainClassifiers();

    return res;
}

bool linearClassifierThread::forgetAll()
{
    stopAll();
    checkKnownObjects();

    for (int i=0; i<knownObjects.size(); i++)
        forgetClass(knownObjects[i].first, false);

    trainClassifiers();
    return true;

}

void linearClassifierThread::stopAll()
{

    //mutex->wait();
    mutex.wait();

    currentState=STATE_DONOTHING;

    if(objFeatures.is_open())
        objFeatures.close();

    //mutex->post();
    mutex.post();

}

int linearClassifierThread::getdir(string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error opening " << dir << endl;
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

void linearClassifierThread::prepareObjPath(string objName)
{

    stopAll();

    //mutex->wait();
    mutex.wait();

    pathObj=currPath+"/"+objName;

    if(yarp::os::stat(pathObj.c_str()))
    {
        createFullPath(pathObj.c_str());
        pathObj=pathObj+"/1.txt";
    }
    else
    {
        char tmpPath[255];
        bool proceed=true;
       
        for (int i=1; proceed; i++)
        {
               sprintf(tmpPath,"%s/%d.txt",pathObj.c_str(),i);
               proceed=!yarp::os::stat(tmpPath);
               sprintf(tmpPath,"%s/%d.txt",pathObj.c_str(),i);
        }

        pathObj=tmpPath;

    }

    objFeatures.open(pathObj.c_str(),fstream::out | fstream::out);
    currentState=STATE_SAVING;

    //mutex->post();
    mutex.post();


}

void linearClassifierThread::createFullPath(const char * path)
{

    if (yarp::os::stat(path))
    {
        string strPath=string(path);
        size_t found=strPath.find_last_of("/");
    
        while (strPath[found]=='/')
            found--;

        createFullPath(strPath.substr(0,found+1).c_str());
        yarp::os::mkdir(strPath.c_str());
    }

}

void linearClassifierThread::interrupt() {

    cout << "begin interrupting ports in thread" << endl;
    //this->commandPort->interrupt();
    this->featuresPort.interrupt();
    this->outputPort.interrupt();
    this->scorePort.interrupt();
    cout << "end interrupting ports in thread" << endl;
}

void linearClassifierThread::threadRelease() 
{

    cout << "begin releasing svm classifiers" << endl;

#ifdef LIBSVMLIN_AVAILABLE

    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
    {
        if(linearClassifiers_SVM.size()>0)
        for (int i=0; i<linearClassifiers_SVM.size(); i++)
            linearClassifiers_SVM[i].freeModel();
    }
#endif

    cout << "begin releasing rls classifiers" << endl;

#ifdef GURLS_AVAILABLE
    
    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
    {
        if(linearClassifiers_RLS.size()>0)
        for (int i=0; i<linearClassifiers_RLS.size(); i++)
            linearClassifiers_RLS[i].freeModel();
    }

#endif

    cout << "begin closing ports in thread" << endl;
    //this->commandPort->close();
    this->featuresPort.close();
    this->outputPort.close();
    this->scorePort.close();
    cout << "end closing ports in thread" << endl;
    //delete mutex;

}

bool linearClassifierThread::setMode(string mode_name)
{
    //mutex->wait();
    mutex.wait();

    if(this->use_only_prediction_mode==BOTH)
        this->prediction_mode = (!strcmp(mode_name.c_str(),"SVM") || !strcmp(mode_name.c_str(),"svm"))?SVM:RLS;
    else
        this->prediction_mode = this->use_only_prediction_mode;

    //mutex->post();
    mutex.post();

    return true;
}

bool linearClassifierThread::getClassList(Bottle &b)
{
    //mutex->wait();
    mutex.wait();

    for(int i=0; i<knownObjects.size(); i++)
        b.addString(knownObjects[i].first.c_str());

    //mutex->post();
    mutex.post();

    return true;
}

void linearClassifierThread::checkKnownObjects()
{

    knownObjects.clear();

#ifdef GURLS_AVAILABLE

    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)
        linearClassifiers_RLS.clear();

#endif


#ifdef LIBSVMLIN_AVAILABLE

    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)
        linearClassifiers_SVM.clear();

#endif

    if(yarp::os::stat(currPath.c_str()))
    {
        createFullPath(currPath.c_str());
        return;
    }

    vector<string> files;
    getdir(currPath,files);

    for (int i=0; i< files.size(); i++)
    {
        if(!files[i].compare(".") || !files[i].compare(".."))
            continue;

#ifdef LIBSVMLIN_AVAILABLE
        if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)
            if(!files[i].compare("svmmodel"))
                continue;
#endif

#ifdef GURLS_AVAILABLE
        if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)
            if(!files[i].compare("rlsmodel"))
                continue;
#endif

        string objPath=currPath+"/"+files[i];
        
#ifdef LIBSVMLIN_AVAILABLE
        if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)  
        {
            string svmPath=objPath+"/svmmodel";
            if(!yarp::os::stat(svmPath.c_str()))
                SVMLinear svm_model(files[i]);
        }
#endif

#ifdef GURLS_AVAILABLE
        if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)
        {
            string rlsPath=objPath+"/rlsmodel";
            if(!yarp::os::stat(rlsPath.c_str()))
                RLSLinear rls_model(files[i]);
        }
#endif

        vector<string> featuresFile;
        vector<string> tmpFiles;
        getdir(objPath,featuresFile);
        
        for (int j=0; j< featuresFile.size(); j++)
        {
            if(!featuresFile[j].compare(".") || !featuresFile[j].compare(".."))
                continue;

            string tmp=objPath+"/"+featuresFile[j];
            tmpFiles.push_back(tmp);
        }

        pair<string, vector<string> > obj(files[i],tmpFiles);
        knownObjects.push_back(obj);

    }

}

void linearClassifierThread::run(){

    int current=0;
    while (!isStopping()) {

        Bottle *p=featuresPort.read(false);

        if(p==NULL)
            continue;

        vector<double> feature;
        feature.resize(p->size());


        for (int i=0; i<p->size(); i++)
            feature[i]=p->get(i).asDouble();

        Stamp timestamp;
        featuresPort.getEnvelope(timestamp);

        //mutex->wait();
        mutex.wait();

        if(currentState==STATE_DONOTHING)
        {   
            true_class = "?";
            //mutex->post();
            mutex.post();
            continue;
        }

        if(currentState==STATE_SAVING)
        {
            for (int i=0; i<feature.size(); i++)
                objFeatures << feature[i] << " ";
            objFeatures << endl;
        }

        if(currentState==STATE_RECOGNIZING)
        {           

#if defined(GURLS_AVAILABLE) && defined(LIBSVMLIN_AVAILABLE)

            if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
            {
                if(linearClassifiers_RLS.size()==0)
                {
                    //mutex->post();
                    mutex.post();
                    continue;
                }
            }
            else
            {
                if(linearClassifiers_SVM.size()==0)
                {
                    //mutex->post();
                    mutex.post();
                    continue;
                }
            }

#elif defined(GURLS_AVAILABLE)

            if(linearClassifiers_RLS.size()==0)
            {
                    //mutex->post();
                    mutex.post();
                continue;
            }

#else

            if(linearClassifiers_SVM.size()==0)
            {
                    //mutex->post();
                    mutex.post();
                continue;
            }

#endif

            //SVM ******************************************************************************

#ifdef LIBSVMLIN_AVAILABLE

            double test_time_SVM;
            string winnerClass_SVM;
            if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
            {
                double init_time=Time::now();
                //cout << "ISTANT SCORES: ";
                double maxVal_SVM=-1000;
                double minValue_SVM=1000;
                double idWin_SVM=-1;

                for(int i =0; i<linearClassifiers_SVM.size(); i++)
                {
                    double value_SVM=linearClassifiers_SVM[i].predictModel(feature);
                    if(value_SVM>maxVal_SVM)
                    {
                        maxVal_SVM=value_SVM;
                        idWin_SVM=i;
                    }
                    if(value_SVM<minValue_SVM)
                        minValue_SVM=value_SVM;

                    bufferScores_SVM[current%bufferSize][i]=(value_SVM);
                    countBuffer_SVM[current%bufferSize][i]=0;
                    //cout << knownObjects[i].first << " " << value << " ";
                }
                countBuffer_SVM[current%bufferSize][idWin_SVM]=1;

                vector<double> avgScores_SVM(linearClassifiers_SVM.size(),0.0);
                vector<double> bufferVotes_SVM(linearClassifiers_SVM.size(),0.0);

                for(int i =0; i<bufferSize; i++)
                    for(int k =0; k<linearClassifiers_SVM.size(); k++)
                    {
                        avgScores_SVM[k]=avgScores_SVM[k]+bufferScores_SVM[i][k];
                        bufferVotes_SVM[k]=bufferVotes_SVM[k]+countBuffer_SVM[i][k];
                    }

                double maxValue_SVM=-100;
                double maxVote_SVM=0;
                int indexClass_SVM=-1;
                int indexMaxVote_SVM=-1;
                
                for(int i =0; i<linearClassifiers_SVM.size(); i++)
                {
                    avgScores_SVM[i]=avgScores_SVM[i]/bufferSize;
                    if(avgScores_SVM[i]>maxValue_SVM)
                    {
                        maxValue_SVM=avgScores_SVM[i];
                        indexClass_SVM=i;
                    }
                    if(bufferVotes_SVM[i]>maxVote_SVM)
                    {
                        maxVote_SVM=bufferVotes_SVM[i];
                        indexMaxVote_SVM=i;
                    }
                    //cout  << knownObjects[i].first << " S: " << avgScores[i] << " V: " << bufferVotes[i] << " " ;
                }


                winnerClass_SVM=knownObjects[indexClass_SVM].first;
                string winnerVote_SVM=knownObjects[indexMaxVote_SVM].first;
                
                if(bufferVotes_SVM[indexMaxVote_SVM]/bufferSize<0.75)
                    winnerClass_SVM="?";

                test_time_SVM=Time::now()-init_time;
            }

#endif

            //RLS ********************************************************************************
            
#ifdef GURLS_AVAILABLE

            double test_time_RLS;
            string winnerClass_RLS;
            if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
            {
                double init_time=Time::now();

                double maxVal_RLS=-1000;
                double minValue_RLS=1000;
                double idWin_RLS=-1;


                gurls::gMat2D<float> feature_RLS(1,feature.size()+1);

                for(int i=0; i<feature.size(); i++)
                    feature_RLS(0,i)=feature[i];
                feature_RLS(0,feature.size())=1.0;

                for(int i =0; i<linearClassifiers_RLS.size(); i++)
                {
                    double value_RLS=linearClassifiers_RLS[i].predictModel(feature_RLS);

                    if(value_RLS>maxVal_RLS)
                    {
                        maxVal_RLS=value_RLS;
                        idWin_RLS=i;
                    }
                    if(value_RLS<minValue_RLS)
                        minValue_RLS=value_RLS;

                    bufferScores_RLS[current%bufferSize][i]=(value_RLS);
                    countBuffer_RLS[current%bufferSize][i]=0;
                    //cout << knownObjects[i].first << " " << value << " ";
                }
                countBuffer_RLS[current%bufferSize][idWin_RLS]=1;

                vector<double> avgScores_RLS(linearClassifiers_RLS.size(),0.0);
                vector<double> bufferVotes_RLS(linearClassifiers_RLS.size(),0.0);

                for(int i =0; i<bufferSize; i++)
                    for(int k =0; k<linearClassifiers_RLS.size(); k++)
                    {
                        avgScores_RLS[k]=avgScores_RLS[k]+bufferScores_RLS[i][k];
                        bufferVotes_RLS[k]=bufferVotes_RLS[k]+countBuffer_RLS[i][k];
                    }

                double maxValue_RLS=-100;
                double maxVote_RLS=0;
                int indexClass_RLS=-1;
                int indexMaxVote_RLS=-1;
                
                for(int i =0; i<linearClassifiers_RLS.size(); i++)
                {
                    avgScores_RLS[i]=avgScores_RLS[i]/bufferSize;
                    if(avgScores_RLS[i]>maxValue_RLS)
                    {
                        maxValue_RLS=avgScores_RLS[i];
                        indexClass_RLS=i;
                    }
                    if(bufferVotes_RLS[i]>maxVote_RLS)
                    {
                        maxVote_RLS=bufferVotes_RLS[i];
                        indexMaxVote_RLS=i;
                    }
                    //cout  << knownObjects[i].first << " S: " << avgScores[i] << " V: " << bufferVotes[i] << " " ;
                }

                winnerClass_RLS=knownObjects[indexClass_RLS].first;
                string winnerVote_RLS=knownObjects[indexMaxVote_RLS].first;
                
                if(bufferVotes_RLS[indexMaxVote_RLS]/bufferSize<0.75)
                    winnerClass_RLS="?";

                test_time_RLS=Time::now()-init_time;
            }

#endif


            if(print_times)
            {
#ifdef GURLS_AVAILABLE
                if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
                    printf("RLS test time: %f\n",test_time_RLS);
#endif

#ifdef LIBSVMLIN_AVAILABLE
                if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
                    printf("SVM test time: %f\n",test_time_SVM);
#endif
            }

            // OUTPUT ************************************************************************
            if(outputPort.getOutputCount()>0)
            {
                Bottle &b=outputPort.prepare();
                b.clear();

#if defined(GURLS_AVAILABLE) && defined(LIBSVMLIN_AVAILABLE)
                if(this->prediction_mode==SVM)
                    b.addString(winnerClass_SVM.c_str());
                else
                    b.addString(winnerClass_RLS.c_str());
#elif defined(GURLS_AVAILABLE)
                b.addString(winnerClass_RLS.c_str());
#else
                b.addString(winnerClass_SVM.c_str());
#endif

                outputPort.write();
            }

            if(scorePort.getOutputCount()>0)
            {
                Bottle allScores;

#if defined(GURLS_AVAILABLE) && defined(LIBSVMLIN_AVAILABLE)

                if(this->prediction_mode==SVM)
                    for(int i =0; i<linearClassifiers_SVM.size(); i++)
                    {
                        Bottle &b=allScores.addList();
                        b.addString(knownObjects[i].first.c_str());
                        b.addDouble(bufferScores_SVM[current%bufferSize][i]);
                    }
                else
                    for(int i =0; i<linearClassifiers_RLS.size(); i++)
                    {
                        Bottle &b=allScores.addList();
                        b.addString(knownObjects[i].first.c_str());
                        b.addDouble(bufferScores_RLS[current%bufferSize][i]);
                    }

#elif defined(GURLS_AVAILABLE)
                for(int i =0; i<linearClassifiers_RLS.size(); i++)
                {
                    Bottle &b=allScores.addList();
                    b.addString(knownObjects[i].first.c_str());
                    b.addDouble(bufferScores_RLS[current%bufferSize][i]);
                }
#else
                for(int i =0; i<linearClassifiers_SVM.size(); i++)
                {
                    Bottle &b=allScores.addList();
                    b.addString(knownObjects[i].first.c_str());
                    b.addDouble(bufferScores_SVM[current%bufferSize][i]);
                }
#endif

                allScores.addString(true_class.c_str());
                scorePort.setEnvelope(timestamp);
                scorePort.write(allScores);
            }

            current++;

        }

        //mutex->post();
        mutex.post();
    }

}

bool linearClassifierThread::startRecognition()
{

    stopAll();

    bool check = true;

#ifdef LIBSVMLIN_AVAILABLE
    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
        if(this->linearClassifiers_SVM.size()==0)
            check = false;
#endif

#ifdef GURLS_AVAILABLE
    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
        if(this->linearClassifiers_RLS.size()==0)
            check = false;
#endif

    //mutex->wait();
    mutex.wait();

    currentState=STATE_RECOGNIZING;

#ifdef LIBSVMLIN_AVAILABLE

    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==SVM)        
    {
        double init_time=Time::now();
        this->countBuffer_SVM.resize(bufferSize);
        this->bufferScores_SVM.resize(bufferSize);
        
        for (int i=0; i<bufferSize; i++)
        {
            bufferScores_SVM[i].resize(linearClassifiers_SVM.size());
            countBuffer_SVM[i].resize(linearClassifiers_SVM.size());
        }
    }

#endif

#ifdef GURLS_AVAILABLE
    
    if(this->use_only_prediction_mode==BOTH || this->use_only_prediction_mode==RLS)        
    {
        this->countBuffer_RLS.resize(bufferSize);
        this->bufferScores_RLS.resize(bufferSize);

        for (int i=0; i<bufferSize; i++)
        {
            bufferScores_RLS[i].resize(linearClassifiers_RLS.size());
            countBuffer_RLS[i].resize(linearClassifiers_RLS.size());
        }    
    }
    
#endif

    //mutex->post();
    mutex.post();

    return true;
}

bool linearClassifierThread::set_true_class(string _true_class)
{
    true_class=_true_class;
    return true;
}
bool linearClassifierThread::get_true_class(string &_true_class)
{
    _true_class=true_class;
    return true;
}