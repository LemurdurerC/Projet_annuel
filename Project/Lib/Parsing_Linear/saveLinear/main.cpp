#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

void test(double *model, int nbFeatures){
    ofstream flux;
    flux.open("MLP_SAVE.txt");
    if (flux)
    {
        flux<<nbFeatures<<endl;
        for (auto i = 0; i < nbFeatures + 1; i++) {
            flux << model[i] << endl;
        }
    }
    else{
        cout << "Erreur : Fichier impossible à ouvrir" << endl;
    }
}

double * lol(void)
{
    ifstream flux("MLP_SAVE.txt");
    if(flux){
        int nb;
        string ligne;
        getline(flux, ligne);
        nb = atoi(ligne.c_str());
        auto w = new double[nb];

        int i=0;
        while(getline(flux, ligne))
        {
            w[i]=atof(ligne.c_str());
            i++;

        }
        return w; // Retourne que le W, donc la taille ne bouge pas dans notre modèle? Constante?
                //Perte à la conversion atoi ect, problème?
    }
    else
    {
        cout << "ERREUR: Impossible d'ouvrir le fichier en lecture." << endl;
        return 0;
    }

}


int main() {
    auto nbFeatures = 9;
    auto w = new double[nbFeatures];
    for (auto i = 0; i < nbFeatures+1; i++) {
        w[i] = ((double) rand()) / RAND_MAX * 2.0 - 1.0;

    }
    test(w,nbFeatures);
    auto w1 = new double[9];
    w1=lol();
    printf("%f",w1[9]);
}
