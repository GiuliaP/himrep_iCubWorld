

#include <iostream>
#include <string>

#include "gurls++/gurls.h"
#include "gurls++/gmat2d.h"

#include <vector>
#include <fstream>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

using namespace std;

class  RLSLinear
{
    private:

        gurls::GURLS RLS;
        string className;
        gurls::GurlsOptionsList* modelLinearRLS;

    public:
        RLSLinear(string className);

        void trainModel(gurls::gMat2D<float> &X, gurls::gMat2D<float> &Y, float lambda=-1.0f);

        float predictModel(gurls::gMat2D<float> &X);

        void saveModel(string pathFile);
        void loadModel(string pathFile);
        void freeModel() {if(modelLinearRLS!=NULL) delete modelLinearRLS;};

};
