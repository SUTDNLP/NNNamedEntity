################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../GRNNCRFMLLabeler.cpp \
../GRNNCRFMMLabeler.cpp \
../GRNNLabeler.cpp \
../GatedCRFMLLabeler.cpp \
../GatedCRFMMLabeler.cpp \
../GatedLabeler.cpp \
../LSTMCRFMLLabeler.cpp \
../LSTMCRFMMLabeler.cpp \
../LSTMLabeler.cpp \
../RNNCRFMLLabeler.cpp \
../RNNCRFMMLabeler.cpp \
../RNNLabeler.cpp \
../SparseCRFMLLabeler.cpp \
../SparseCRFMMLabeler.cpp \
../SparseGRNNCRFMLLabeler.cpp \
../SparseGRNNCRFMMLabeler.cpp \
../SparseGRNNLabeler.cpp \
../SparseGatedCRFMLLabeler.cpp \
../SparseGatedCRFMMLabeler.cpp \
../SparseGatedLabeler.cpp \
../SparseLSTMCRFMLLabeler.cpp \
../SparseLSTMCRFMMLabeler.cpp \
../SparseLSTMLabeler.cpp \
../SparseLabeler.cpp \
../SparseRNNCRFMLLabeler.cpp \
../SparseRNNCRFMMLabeler.cpp \
../SparseRNNLabeler.cpp \
../SparseTNNCRFMLLabeler.cpp \
../SparseTNNCRFMMLabeler.cpp \
../SparseTNNLabeler.cpp \
../TNNCRFMLLabeler.cpp \
../TNNCRFMMLabeler.cpp \
../TNNLabeler.cpp 

OBJS += \
./GRNNCRFMLLabeler.o \
./GRNNCRFMMLabeler.o \
./GRNNLabeler.o \
./GatedCRFMLLabeler.o \
./GatedCRFMMLabeler.o \
./GatedLabeler.o \
./LSTMCRFMLLabeler.o \
./LSTMCRFMMLabeler.o \
./LSTMLabeler.o \
./RNNCRFMLLabeler.o \
./RNNCRFMMLabeler.o \
./RNNLabeler.o \
./SparseCRFMLLabeler.o \
./SparseCRFMMLabeler.o \
./SparseGRNNCRFMLLabeler.o \
./SparseGRNNCRFMMLabeler.o \
./SparseGRNNLabeler.o \
./SparseGatedCRFMLLabeler.o \
./SparseGatedCRFMMLabeler.o \
./SparseGatedLabeler.o \
./SparseLSTMCRFMLLabeler.o \
./SparseLSTMCRFMMLabeler.o \
./SparseLSTMLabeler.o \
./SparseLabeler.o \
./SparseRNNCRFMLLabeler.o \
./SparseRNNCRFMMLabeler.o \
./SparseRNNLabeler.o \
./SparseTNNCRFMLLabeler.o \
./SparseTNNCRFMMLabeler.o \
./SparseTNNLabeler.o \
./TNNCRFMLLabeler.o \
./TNNCRFMMLabeler.o \
./TNNLabeler.o 

CPP_DEPS += \
./GRNNCRFMLLabeler.d \
./GRNNCRFMMLabeler.d \
./GRNNLabeler.d \
./GatedCRFMLLabeler.d \
./GatedCRFMMLabeler.d \
./GatedLabeler.d \
./LSTMCRFMLLabeler.d \
./LSTMCRFMMLabeler.d \
./LSTMLabeler.d \
./RNNCRFMLLabeler.d \
./RNNCRFMMLabeler.d \
./RNNLabeler.d \
./SparseCRFMLLabeler.d \
./SparseCRFMMLabeler.d \
./SparseGRNNCRFMLLabeler.d \
./SparseGRNNCRFMMLabeler.d \
./SparseGRNNLabeler.d \
./SparseGatedCRFMLLabeler.d \
./SparseGatedCRFMMLabeler.d \
./SparseGatedLabeler.d \
./SparseLSTMCRFMLLabeler.d \
./SparseLSTMCRFMMLabeler.d \
./SparseLSTMLabeler.d \
./SparseLabeler.d \
./SparseRNNCRFMLLabeler.d \
./SparseRNNCRFMMLabeler.d \
./SparseRNNLabeler.d \
./SparseTNNCRFMLLabeler.d \
./SparseTNNCRFMMLabeler.d \
./SparseTNNLabeler.d \
./TNNCRFMLLabeler.d \
./TNNCRFMMLabeler.d \
./TNNLabeler.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


