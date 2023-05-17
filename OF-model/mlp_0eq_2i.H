/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2020 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::RASModels::mlp_0eq_2i

Description
    Zero-Equation deep learning driven turbulence model that directly computes the turbulent viscosity with a simple relation depending on magnitude of the velocity and a length scale (mininum distance to a wall). Works for indoor airflow. The turbulent viscosity is computed by 

    Reference for zero equation model:
    \verbatim
        Chen, Qingyan, and Weiran Xu. "A zero-equation turbulence model for indoor airflow simulation." Energy and buildings 28.2 (1998): 137-144.

SourceFiles
    mlp_0eq_2i.C

\*---------------------------------------------------------------------------*/

#ifndef mlp_0eq_2i_H
#define mlp_0eq_2i_H

#include "RASModel.H"
#include "eddyViscosity.H"
#include <tensorflow/c/c_api.h>
#include <cstddef>
#include <cstdint>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
    Class mlp_0eq_2i Declaration
    \*---------------------------------------------------------------------------*/

template<class BasicMomentumTransportModel>
class mlp_0eq_2i
:
    public eddyViscosity<RASModel<BasicMomentumTransportModel>>
{
    // Private Member Functions

        // Disallow default bitwise copy construct and assignment
    mlp_0eq_2i(const mlp_0eq_2i&);
    void operator=(const mlp_0eq_2i&);


protected:

    // Protected data

    // Model coefficients
    const string ANNmodelDir_;
    TF_Graph* Graph;
    TF_Status* Status;
    TF_SessionOptions* SessionOpts;
    TF_Session* Session; 
    TF_Buffer* RunOpts;
    int NumInputs;
    TF_Output* Input;
    TF_Output t0;
    int NumOutputs;
    TF_Output* Output;
    TF_Tensor** InputValues;
    TF_Tensor** OutputValues;
    int num_cells;
    const char* saved_model_dir;
    // Fields
    const volScalarField& y_;
    // Protected Member Functions
    virtual void correctNut();
    // virtual void loadMLP();
    void run_mlp(int num_inputs, int num_outputs);
    static void NoOpDeallocator(void* data, size_t a, void* b) {};
    static float Clamp( float value, float min, float max ) {
        return (value < min) ? min : (value > max) ? max : value;
    };
    
public:

    typedef typename BasicMomentumTransportModel::alphaField alphaField;
    typedef typename BasicMomentumTransportModel::rhoField rhoField;
    typedef typename BasicMomentumTransportModel::transportModel transportModel;


    //- Runtime type information
    TypeName("mlp_0eq_2i");

    //- Construct from components
    mlp_0eq_2i
    (
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& type = typeName
    );

    //- Destructor
    virtual ~mlp_0eq_2i()
        {}
    
    // Member Functions
    //- Re-read model coefficients if they have changed
    virtual bool read();


    //FROM SPALART ALLMARAS
    virtual tmp<volScalarField> k() const;
    //FROM SPALART ALLMARAS
    virtual tmp<volScalarField> epsilon() const;

    //- Solve the turbulence equations and correct the turbulence viscosity
    virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
#include "mlp_0eq_2i.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
