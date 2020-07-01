#include <iostream>
#include <random>

struct ClusterRepresentative{
    double *formerCoord;
    double *coord;

    double **memberOfCluster;
    int countClusterMember;

    bool finalPlace;

    void initialise(int,int);
};


void ClusterRepresentative::initialise(int dataset_samples_count,int dataset_sample_features_count){

    formerCoord = new double[dataset_sample_features_count];
    coord = new double[dataset_sample_features_count];

    for(int i = 0; i<dataset_sample_features_count;i++) {
        std::random_device rd;
        std::mt19937 mt(rd());
        //POTENTIAL ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        std::uniform_int_distribution<int> dist(5, 15);
        auto randNumber = dist(mt);
        coord[i] = randNumber;
        formerCoord[i] = coord[i];
    }

    memberOfCluster = new double*[dataset_samples_count];
    for(int i = 0; i<dataset_samples_count;i++){
        memberOfCluster[i] = new double[dataset_sample_features_count];
    }

    countClusterMember = 0;
    finalPlace = false;
}




//juste for test in C++
double *getPartsOfTab(int start, int stop, double *tab) {
    //start and stop are in newTab (inclus)
    auto newTab = new double[(stop - start) + 1];

    int j = 0;
    for (int i = start; i <= stop; i++) {
        newTab[j] = tab[i];
        j++;
    }

    return newTab;

}



//POTENTIAL ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
double distanceBetween2Points(double *coordsA, double *coordsB, int numberOfCoordsAandB){
    double result = 0.0;

    for(int i = 0; i<numberOfCoordsAandB;i++){
        result += fabs(coordsB[i] - coordsA[i]);
    }

    return result;
}


bool checkEquality2Points(double *coordsA, double *coordsB, int numberOfCoordsAandB){
    for(int i = 0; i<numberOfCoordsAandB;i++) {
        if(coordsA[i] != coordsB[i]){
            return false;
        }
    }
    return true;
}


void copyCoordAtoCoordB(double *coordsA, double *coordsB, int numberOfCoordsAandB){
    for(int i = 0; i<numberOfCoordsAandB;i++) {
        coordsB[i] = coordsA[i];
    }
}

void initialiseTabTo0(double *tab,int tabSize){
    for(int i = 0;i<tabSize;i++){
        tab[i] = 0.0;
    }
}


void showPoint(double *coords, int sizeCoord){
    printf("(");
    for(int i =0;i<sizeCoord;i++){
        printf("%f ",coords[i]);
    }
    printf(")\n");
}


void addTabAtoTabB(double *coordsA, double *coordsB, int numberOfCoordsAandB){

    for(int i =0; i<numberOfCoordsAandB;i++){
        coordsA[i] += coordsB[i];
    }
}


//Algo de LLoyd
/*
Parameter :
 1) Number of cluster ->  number of representant
 2) Dataset

Step :
 1) Cluster (representant)
    -> Place (or replace) cluster
 2) Each point
    -> Find nearest cluster
    -> Assing to this cluster
*/



ClusterRepresentative** algoOfLLoyd(int numberOfCluster, double *dataset, int dataset_samples_count, int dataset_sample_features_count){
    ClusterRepresentative **tabCluster = new ClusterRepresentative*[numberOfCluster];

    for(int i = 0; i<numberOfCluster;i++){
        ClusterRepresentative *c = new ClusterRepresentative();
        c->initialise(dataset_samples_count,dataset_sample_features_count);
        showPoint(c->coord,dataset_sample_features_count);
        tabCluster[i] = c;
    }
    bool stop = false;


    while(stop == false) {


        //Set all to 0 except coordinate of cluster
        for(int i = 0; i<numberOfCluster;i++) {
            tabCluster[i]->memberOfCluster = new double*[dataset_samples_count];
            for(int j = 0; j<dataset_samples_count;j++){
                tabCluster[i]->memberOfCluster[j] = new double[dataset_sample_features_count];
            }
            tabCluster[i]->countClusterMember = 0;

        }


        //ALGO STEP 1 : check each point in dataset
        for (int i = 0;
             i + dataset_sample_features_count <= dataset_samples_count*dataset_sample_features_count;
             i = i + dataset_sample_features_count) {

            //POTENTIAL ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            double distMin = 1000000000000.0;
            int theCluster = -1;
            for (int j = 0; j < numberOfCluster; j++) {
                //find the nearest cluster
                //just for C++ test
                double *coordsA = getPartsOfTab(i, i + dataset_sample_features_count - 1, dataset);
                if (distanceBetween2Points(coordsA, tabCluster[j]->coord, dataset_sample_features_count) < distMin) {
                    distMin = distanceBetween2Points(coordsA, tabCluster[j]->coord, dataset_sample_features_count);
                    theCluster = j;
                }


            }

            //assign the inputs to the right cluster
            double *dataInCluster = getPartsOfTab(i, i + dataset_sample_features_count - 1, dataset);

            tabCluster[theCluster]->memberOfCluster[tabCluster[theCluster]->countClusterMember] = dataInCluster;
            tabCluster[theCluster]->countClusterMember++;


        }


        //ALGO STEP 2 : check cluster
        for (int i = 0; i < numberOfCluster; i++) {
            printf("%Il ya %d membres\n",tabCluster[i]->countClusterMember);
            if(!tabCluster[i]->finalPlace && (tabCluster[i]->countClusterMember > 0)) {
                copyCoordAtoCoordB(tabCluster[i]->coord, tabCluster[i]->formerCoord, dataset_sample_features_count);

                //mean of all point assigned to cluster i
                double *newCoord = new double[dataset_sample_features_count];
                initialiseTabTo0(newCoord, dataset_sample_features_count);

                for (int j = 0; j < tabCluster[i]->countClusterMember; j++) {
                    addTabAtoTabB(newCoord, tabCluster[i]->memberOfCluster[j], dataset_sample_features_count);
                }

                for (int k = 0; k < dataset_sample_features_count; k++) {
                    newCoord[k] = newCoord[k] / tabCluster[i]->countClusterMember;
                }

                copyCoordAtoCoordB(newCoord, tabCluster[i]->coord, dataset_sample_features_count);

                if (checkEquality2Points(tabCluster[i]->formerCoord, tabCluster[i]->coord,
                                         dataset_sample_features_count)) {
                    tabCluster[i]->finalPlace = true;
                    break;
                }

                /* }else if(tabCluster[i]->countClusterMember == 0){
                     tabCluster[i]->finalPlace = true;
                 }*/
            }

        }
        printf("\n");





        //ALGO STOP check if we stop
        int finishCluster = 0;
        for (int i = 0; i < numberOfCluster; i++) {
            if (tabCluster[i]->finalPlace) {
                finishCluster++;
            }
        }
        if (finishCluster == numberOfCluster) {
            stop = true;
        }




    }

    printf("FINISH\n");

    return tabCluster;
}



void disposeAllCluster(ClusterRepresentative **allCluster,int numberOfCluster){
    for(int i = 0; i<numberOfCluster;i++){
        delete allCluster[i];
    }
    delete allCluster;
}



int main() {
    std::cout << "Hello, World!" << std::endl;


    const int nbreFeature = 2;
    const int nbreEnter  = 15;

    int numberOfCluser = 3;

    double X[nbreFeature*nbreEnter] = {
                    1.0,3.0,
                    1.0,4.0,
                    2.0,2.0,
                    2.0,5.0,
                    3.0,3.0,

                    6.0,7.0,
                    7.0,8.0,
                    8.0,9.0,
                    8.0,5.0,
                    9.0,8.0,

                    13.0,13.0,
                    13.0,14.0,
                    15.0,18.0,
                    19.0,11.0,
                    16.0,17.0

                    };


    ClusterRepresentative **tabCluster = algoOfLLoyd(numberOfCluser,X,nbreEnter,nbreFeature);

    for(int i = 0;i<numberOfCluser;i++){
        printf("( ");
        for(int j = 0; j<nbreFeature;j++) {
            printf("%f ", tabCluster[i]->coord[j]);
        }
        printf(")");
        printf("\n");

    }


    disposeAllCluster(tabCluster,numberOfCluser);



    return 0;
}
