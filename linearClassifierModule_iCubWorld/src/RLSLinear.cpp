#include "gurls++/gurls.h"
#include "gurls++/primal.h"

#include "RLSLinear.h"

using namespace gurls;

RLSLinear::RLSLinear(string className) {

    this->className=className;
    modelLinearRLS=NULL;

}

void RLSLinear::trainModel(gMat2D<float> &X, gMat2D<float> &Y, float lambda)
{
    if(modelLinearRLS!=NULL)
        delete modelLinearRLS;

    // specify the task sequence and processes
    OptTaskSequence *seq = new OptTaskSequence();

    // defines instructions for training process
    OptProcess* process_train = new OptProcess();
    OptProcess* process_predict = new OptProcess();

    if(lambda<0)
    {
        *seq << "kernel:linear" << "split:ho" << "paramsel:hodual";
        *process_train << GURLS::computeNsave << GURLS::compute << GURLS::computeNsave;
        *process_predict << GURLS::load << GURLS::ignore << GURLS::load;
    }

    *seq<< "optimizer:rlsdual"<< "pred:dual";
    *process_train<<GURLS::computeNsave<<GURLS::ignore;
    *process_predict<<GURLS::load<<GURLS::computeNsave;

    GurlsOptionsList * processes = new GurlsOptionsList("processes", false);
    processes->addOpt("train",process_train);
    processes->addOpt("test",process_predict);

    // build the options list
    modelLinearRLS = new GurlsOptionsList(className, true);

    modelLinearRLS->addOpt("seq", seq);
    modelLinearRLS->addOpt("processes", processes);

    // train!
    RLS.run(X,Y,*modelLinearRLS,"train");

}


void RLSLinear::saveModel(string pathFile)
{
//    save_model(pathFile.c_str(), modelLinearSVM);
}

void RLSLinear::loadModel(string pathFile)
{
//    modelLinearSVM=load_model(pathFile.c_str());
}

float RLSLinear::predictModel(gMat2D<float> &X)
{
    
    if(modelLinearRLS==NULL)
    {
        fprintf(stdout,"[RLS] Error, Train Model First \n");
        return 0.0;
    }
    
    gMat2D<float> empty;

    RLS.run(X,empty,*modelLinearRLS,"test");

    gMat2D<float>& pred = modelLinearRLS->getOptValue<OptMatrix<gMat2D<float> > >("pred");
    
    return pred(0,0);
}


