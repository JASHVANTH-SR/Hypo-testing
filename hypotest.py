import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats  
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.random import randn
from statsmodels.stats.weightstats import ztest

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

try:
    os.mkdir("temp")
except:
    pass

st.title('''Hypothesis Testing''')

def load_csv():
        csv = pd.read_csv('AttendanceMarksSA.csv')
        return csv


df=load_csv()
df.hist()
st.pyplot()
Y=df['Attendance']
X1=df['MSE']
X2=df['ESE']
X1 = sm.add_constant(X1)
corr=df.corr()
st.markdown("""## Our Dataset""")
st.write(df)
st.markdown("""### Details of our DataSet""")
st.write(df.describe())
st.markdown("""### Correlation Details""")
st.write(corr)

model1 = sm.OLS(Y, X1, missing='drop')
model1_result = model1.fit()
x1_train,x1_test,y_train,y_test=train_test_split(X1,Y,test_size=0.2,random_state=0)
st.markdown("""### Summary of our ML Model-1 MSE""")
st.write(model1_result.summary())
st.write(model1_result.fittedvalues)

X2 = sm.add_constant(X2)

model2 = sm.OLS(Y, X2, missing='drop')
model2_result = model2.fit()
x2_train,x2_test,y_train,y_test=train_test_split(X2,Y,test_size=0.2,random_state=0)
st.markdown("""### Summary of our ML Model-2 ESE""")
st.write(model2_result.summary())
sns.histplot(model2_result.resid);
st.write(model2_result.fittedvalues)

st.markdown("### Histogram of residuals  1")
sns.histplot(model1_result.resid)
plt.show()
st.pyplot()

st.markdown("### Histogram of residuals Model 2")
sns.histplot(model2_result.resid)
st.pyplot()

Y_max = Y.max()
Y_min = Y.min()

st.markdown("""### kernel density plot and overlay the normal curve with the same mean and standard deviation""")

fig, ax = plt.subplots()
# plot the residuals
from scipy import stats
mu1, std1 = stats.norm.fit(model1_result.resid)
sns.histplot(x=model1_result.resid, ax=ax, stat="density", linewidth=0, kde=True)
ax.set(title="Distribution of residuals", xlabel="residual")

# plot corresponding normal curve
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x1 = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm.pdf(x1, mu1, std1) # calculate the y values for the normal curve
sns.lineplot(x=x1, y=p, color="orange", ax=ax)
plt.show()
st.pyplot()

mu2, std2 = stats.norm.fit(model2_result.resid)
sns.histplot(x=model2_result.resid, ax=ax, stat="density", linewidth=0, kde=True)
ax.set(title="Distribution of residuals", xlabel="residual")

# plot corresponding normal curve
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x2 = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm.pdf(x2, mu2, std2) # calculate the y values for the normal curve
sns.lineplot(x=x2, y=p, color="red", ax=ax)
plt.show()
st.pyplot()



ax = sns.scatterplot(x=model1_result.fittedvalues, y=Y)
ax.set(ylim=(Y_min, Y_max))
ax.set(xlim=(Y_min, Y_max))
ax.set_xlabel("Predicted value of Strength")
ax.set_ylabel("Observed value of Strength")

X_ref = Y_ref = np.linspace(Y_min, Y_max, 100)
plt.plot(X_ref, Y_ref, color='red', linewidth=1)
st.pyplot(plt.show())

st.markdown("""### Boxplot for residuals Model 1""")
sns.boxplot(x=model2_result.resid, showmeans=True);
st.pyplot()

st.markdown("""### Boxplot for residuals Model 2""")
sns.boxplot(x=model2_result.resid, showmeans=True);
st.pyplot()

st.markdown("""### Q-Q Plot Model 1""")
sm.qqplot(model1_result.resid, line='s')
st.pyplot()

st.markdown("""### Q-Q Plot Model 2""")
sm.qqplot(model2_result.resid, line='s')
st.pyplot()

st.markdown("""### Fit Model Plot for Model 1""")
sm.graphics.plot_fit(model1_result,1, vlines=False);
st.pyplot()

st.markdown("""### Fit Model Plot for Model 2""")
sm.graphics.plot_fit(model2_result,1, vlines=False);
st.pyplot()



# Generate a random array of 50 numbers having mean 110 and sd 15
# similar to the IQ scores data we assume above

st.markdown("### Z-Test for Hypothesis Testing MSE")
data1 = df['MSE']
data2= df['ESE']
alpha =0.05
null_mean =100
# print mean and sd
st.write('mean=%.2f stdv=%.2f' % (np.mean(data1), np.std(data1)))
ztest_Score, p_value= ztest(data1,value = null_mean, alternative='larger')
if(p_value < alpha):
	st.write("Reject Null Hypothesis")
else:
	st.write("Fail to Reject NUll Hypothesis")

st.markdown("### Z-Test for Hypothesis Testing ESE")
st.write('mean=%.2f stdv=%.2f' % (np.mean(data2), np.std(data2)))
ztest_Score, p_value= ztest(data2,value = null_mean, alternative='larger')
if(p_value < alpha):
	st.write("Reject Null Hypothesis")
else:
	st.write("Fail to Reject NUll Hypothesis")
