# Python 2.7.10
# sklearn version 0.16.1
import joblib

# Read in energy balance random forest
rf_F = joblib.load('C:/Users/explo/Documents/CAM5 Dataset/multi-resolution/fnet_files/energy_balance_random_forest.pkl') 
# Read in precipitation random forest
rf_P = joblib.load('./p_files/precipitation_random_forest.pkl')

# vector of 11 parameters and resolution. Parameters are scaled (0-1) and
# resolution is 1., 0.25 and 0.0625 for low, medium, high respectively
x = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.0625]

# predict energy balance for parameter values and resolution specified in x
prediction_F = rf_F.predict(x)
print ("energy balance prediction = ", prediction_F)

# predict energy balance for parameter values and resolution specified in x
prediction_P = rf_P.predict(x)
print ("precipitation prediction = ", prediction_P)

# expected results are:
# prediction_F = -5.64980687, 
# prediction_P = 3.00483487 

