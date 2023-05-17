/*---------------------------------------------------------------------------*\
    =========                 |
    \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2021 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/

#include "NNZeroEquation.H"
#include "fvModels.H"
#include "fvConstraints.H"
#include "bound.H"
#include "wallDist.H"

#include <tensorflow/c/c_api.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <vector>
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicMomentumTransportModel>
void NNZeroEquation<BasicMomentumTransportModel>::correctNut()
{
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicMomentumTransportModel>
NNZeroEquation<BasicMomentumTransportModel>::NNZeroEquation
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& type
)
:
    eddyViscosity<RASModel<BasicMomentumTransportModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport
    ),

    // Define location path of the NN saved model 
    ANNmodelDir_
    (
        IOdictionary
        (
            IOobject
            (
                IOobject::groupName("momentumTransport", U.group()),
                this->runTime_.constant(),
                this->mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        ).lookup("ANNmodelDir")
    ),

    // compute wall distance
    y_(wallDist::New(this->mesh_).y())
    
{

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
    
    // obtain information on the location path
    saved_model_dir = ANNmodelDir_.c_str();

    Graph = TF_NewGraph();
    Status = TF_NewStatus();
    SessionOpts = TF_NewSessionOptions();
    RunOpts = NULL;
    NumInputs = 1; 
    
    const char* tags = "serve";
    int ntags = 1;

    // Initialize Tensorflow session and tensors
    Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    Input = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NumInputs));
    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_norm_input"), 0};
    Input[0] = t0;
    
    NumOutputs = 1;
    Output = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NumOutputs));
    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    Output[0] = t2;

    InputValues  = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*)*NumInputs));
    OutputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*)*NumOutputs));
    num_cells = this->mesh_.cells().size();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicMomentumTransportModel>
bool NNZeroEquation<BasicMomentumTransportModel>::read()
{
    if (eddyViscosity<RASModel<BasicMomentumTransportModel>>::read())
    {
        return true;
    }     else
    {
        return false;
    }

}

template<class BasicMomentumTransportModel>
tmp<volScalarField> NNZeroEquation<BasicMomentumTransportModel>::k() const
{
    return tmp<volScalarField>
        (
            new volScalarField
            (
                IOobject
                (
                    "k",
                    this->runTime_.timeName(),
                    this->mesh_
                ),
                this->mesh_,
                dimensionedScalar("0", dimensionSet(0, 2, -2, 0, 0), 0)
            )
        );
}

template<class BasicMomentumTransportModel>
tmp<volScalarField> NNZeroEquation<BasicMomentumTransportModel>::epsilon() const
{
    WarningInFunction
        << "Turbulence kinetic energy dissipation rate not defined for "
            << "Spalart-Allmaras model. Returning zero field"
            << endl;

    return tmp<volScalarField>
        (
            new volScalarField
            (
                IOobject
                (
                    "epsilon",
                    this->runTime_.timeName(),
                    this->mesh_
                ),
                this->mesh_,
                dimensionedScalar("0", dimensionSet(0, 2, -3, 0, 0), 0)
            )
        );
}

template<class BasicMomentumTransportModel>
void NNZeroEquation<BasicMomentumTransportModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }
    eddyViscosity<RASModel<BasicMomentumTransportModel>>::correct();

    int num_inputs = 2;
    int num_outputs = 1;
    run_NN(num_inputs, num_outputs);
}

template<class BasicMomentumTransportModel>void NNZeroEquation<BasicMomentumTransportModel>::run_NN(int num_inputs, int num_outputs)
{
    static char *new_environment = strdup("TF_CPP_MIN_LOG_LEVEL=0");
    putenv(new_environment);

    volScalarField nut_NN = this->nut_;
    float input_vals[num_cells][num_inputs];
    const std::vector<std::int64_t> input_dims = {num_cells, num_inputs};
    
    forAll(this->U_.internalField(), id)
    {
        float i1 = mag(this->U_[id]);
        float i2 = y_[id];
        input_vals[id][0] = i1;
        input_vals[id][1] = i2;
    }
    
    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, input_dims.data(), input_dims.size(), &input_vals, num_cells*num_inputs*sizeof(float), &NoOpDeallocator, 0);
    InputValues[0] = int_tensor;

    // run session to obtain prediction
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL , Status);
    float* final_output = static_cast<float*>(TF_TensorData(OutputValues[0]));

    // limit for stability and physical results between 0 and 100
    forAll(nut_NN.internalField(), id)
    {
        nut_NN[id] = Clamp(final_output[num_outputs*id], 0.0, 100.0);
    }
    
    // assign turbulent viscosity
    this->nut_ = nut_NN;
    this->nut_.correctBoundaryConditions();
    fvConstraints::New(this->mesh_).constrain(this->nut_);

    // Release the memory
    TF_DeleteTensor(int_tensor);
    TF_DeleteTensor(OutputValues[0]);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
