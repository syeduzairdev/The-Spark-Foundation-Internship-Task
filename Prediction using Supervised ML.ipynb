{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f5e6672b",
   "metadata": {},
   "source": [
    "# The Sparks Foundation: Data Science & Business Analytics Internship"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0280aa38",
   "metadata": {},
   "source": [
    "# Name: SYED UZAIR HASSAN"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cfd2a8d",
   "metadata": {},
   "source": [
    "# Prediction Using Supervised Machine Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e5e20d09",
   "metadata": {},
   "source": [
    "# Task # 1: What will be predicted score if a student studies for 9.25 hrs/ day?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a50ab0a",
   "metadata": {},
   "source": [
    "#  Here we Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9defd6fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac9c57fd",
   "metadata": {},
   "source": [
    "# Read and View DataFrame using URL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fdff995c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1.5</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>9.2</td>\n",
       "      <td>88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>5.5</td>\n",
       "      <td>60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8.3</td>\n",
       "      <td>81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2.7</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>7.7</td>\n",
       "      <td>85</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>5.9</td>\n",
       "      <td>62</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>4.5</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>3.3</td>\n",
       "      <td>42</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>1.1</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>8.9</td>\n",
       "      <td>95</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>2.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>1.9</td>\n",
       "      <td>24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>6.1</td>\n",
       "      <td>67</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>7.4</td>\n",
       "      <td>69</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>2.7</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>4.8</td>\n",
       "      <td>54</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>3.8</td>\n",
       "      <td>35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>6.9</td>\n",
       "      <td>76</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>7.8</td>\n",
       "      <td>86</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Hours  Scores\n",
       "0     2.5      21\n",
       "1     5.1      47\n",
       "2     3.2      27\n",
       "3     8.5      75\n",
       "4     3.5      30\n",
       "5     1.5      20\n",
       "6     9.2      88\n",
       "7     5.5      60\n",
       "8     8.3      81\n",
       "9     2.7      25\n",
       "10    7.7      85\n",
       "11    5.9      62\n",
       "12    4.5      41\n",
       "13    3.3      42\n",
       "14    1.1      17\n",
       "15    8.9      95\n",
       "16    2.5      30\n",
       "17    1.9      24\n",
       "18    6.1      67\n",
       "19    7.4      69\n",
       "20    2.7      30\n",
       "21    4.8      54\n",
       "22    3.8      35\n",
       "23    6.9      76\n",
       "24    7.8      86"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url='http://bit.ly/w-data'\n",
    "dataframe=pd.read_csv(url)\n",
    "dataframe"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2710ffa9",
   "metadata": {},
   "source": [
    "# It displays the first 5 rows of the DataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "995dff54",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataframe.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "65da84bb",
   "metadata": {},
   "source": [
    "# It displays the last 5 rows of the DataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "a49bd0e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>2.7</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>4.8</td>\n",
       "      <td>54</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>3.8</td>\n",
       "      <td>35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>6.9</td>\n",
       "      <td>76</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>7.8</td>\n",
       "      <td>86</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Hours  Scores\n",
       "20    2.7      30\n",
       "21    4.8      54\n",
       "22    3.8      35\n",
       "23    6.9      76\n",
       "24    7.8      86"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "749dd3e2",
   "metadata": {},
   "source": [
    "# It displays the total count of rows & columns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "5045f410",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(25, 2)"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23cc2687",
   "metadata": {},
   "source": [
    "# To View Basic Statistical Details"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "11f78564",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25.000000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.012000</td>\n",
       "      <td>51.480000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.525094</td>\n",
       "      <td>25.286887</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.100000</td>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.700000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.800000</td>\n",
       "      <td>47.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.400000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>95.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours     Scores\n",
       "count  25.000000  25.000000\n",
       "mean    5.012000  51.480000\n",
       "std     2.525094  25.286887\n",
       "min     1.100000  17.000000\n",
       "25%     2.700000  30.000000\n",
       "50%     4.800000  47.000000\n",
       "75%     7.400000  75.000000\n",
       "max     9.200000  95.000000"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8cdec36c",
   "metadata": {},
   "source": [
    "# To View Concise Summary of a DataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "c00e3a26",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 528.0 bytes\n"
     ]
    }
   ],
   "source": [
    "dataframe.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3763fb28",
   "metadata": {},
   "source": [
    "# Visualizing the Data Sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "5a0c96a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1800x1080 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "x='Hours'\n",
    "y='Scores'\n",
    "dataframe.plot(x,y,style='o',markerfacecolor='m',mec='m',ms=12)\n",
    "\n",
    "plt.title('Hours vs Percentage',fontdict=font1)\n",
    "plt.xlabel('Hours studied',fontdict=font2)\n",
    "plt.ylabel('Percentage Score',fontdict=font2)\n",
    "plt.grid()\n",
    "plt.legend(loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd392251",
   "metadata": {},
   "source": [
    "# Splitting the Dataset into the Independent Feature Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "0c086e65",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=dataframe.iloc[:, :-1].values   "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5e67d08c",
   "metadata": {},
   "source": [
    "# Extracting the Dataset to Get the Dependent Vector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "f32d41c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "y=dataframe.iloc[:, 1].values    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7c87947b",
   "metadata": {},
   "source": [
    "# Splitting model into training and testing set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "df0eed66",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split \n",
    "X_train, X_test, y_train, y_test= train_test_split(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "185e0a4f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[2.5]\n",
      " [5.1]\n",
      " [8.5]\n",
      " [3.3]\n",
      " [7.7]\n",
      " [6.9]\n",
      " [7.8]\n",
      " [5.9]\n",
      " [1.1]\n",
      " [3.8]\n",
      " [7.4]\n",
      " [3.2]\n",
      " [4.8]\n",
      " [8.9]\n",
      " [6.1]\n",
      " [8.3]\n",
      " [2.7]\n",
      " [4.5]] [30 47 75 42 85 76 86 62 17 35 69 27 54 95 67 81 30 41]\n"
     ]
    }
   ],
   "source": [
    "print(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "183bcac0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.5]\n",
      " [6.9]\n",
      " [1.9]\n",
      " [2.5]\n",
      " [6.1]\n",
      " [1.1]\n",
      " [2.7]] [20 76 24 30 67 17 25]\n"
     ]
    }
   ],
   "source": [
    "print(X_test,y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b7b0654",
   "metadata": {},
   "source": [
    "# Training the Algorithm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "c968a2ca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression  \n",
    "linreg = LinearRegression()\n",
    "\n",
    "linreg.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06c4b023",
   "metadata": {},
   "source": [
    "# Plotting the regression line (y=mx+c) and Training Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "a3358d49",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "line = linreg.coef_*X+linreg.intercept_\n",
    " \n",
    "plt.rcParams['figure.figsize'] = [16, 9]\n",
    "\n",
    "plt.scatter(X_train, y_train,color='blue')\n",
    "\n",
    "plt.plot(X, line,color='green');\n",
    "\n",
    "\n",
    "plt.title('Training Set',fontdict=font1)\n",
    "plt.xlabel('Hours studied',fontdict=font2)\n",
    "plt.ylabel('Percentage Score',fontdict=font2)\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "de9009df",
   "metadata": {},
   "source": [
    "# Plotting for the testing data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "a8ad1dd7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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+C077/zKgjpqbkVfoZUkSZKklrdmzWNcffWuFbV99jmH/fb7UkEdtQYDrSRJkiQNksz1zJ5dGbtGjz6SKVMWFNRRaxlWdAOSJEmS1IquumpcVZidPr27ccLszJkwcSIMG1Z6nTmz6I7q5hVaSZIkSRpAd975HpYsOb+idtJJTzNixHYFdVTDzJlw1lmwalVp+4EHStsAp59eXF91MtBKkiRJ0gBYuvRnLFz41oralCm3MXr0oQV1tAnnnPN8mN1g1apS3UArSZIkSe1h5crbuf76wypqhxzyM8aN+6uCOuqHxYvrqzcoA60kSZIkbYZ1657hyiu3r6iNH38WBx/8vYI6qsM++5RuM65VbyIGWkmSJEmqQ2Yye3bl+rpbbbUrJ564rKCONsO551bOoQUYNapUbyIGWkmSJEnqp+uvP5qVKytXKZ4+fR0RwwvqaDNtmCd7zjml24z32acUZpto/iwYaCVJkiTpBd1336dZvLjy6uUJJyxn5MhdCupoAJx+etMF2N4MtJIkSZLUh8cf/y233PLqitqxx17L9tsfV1BH6slAK0mSJEm9rF69iGuv3beiduCB/8aee76voI5Ui4FWkiRJksrWr3+WK67YtqK2yy5v4PDDLymoI22KgVaSJEmSgM7OqKp1dGQBnai/DLSSJEmS2tqCBS9nxYo/VtSmTXuOYcNGFtSR+stAK0mSJKktPfjgN7n33r+rqB1//INss81eBXWkehloJUmSJLWVJ5+8kptuOrmidtRR/8dOO51SUEfaXAZaSZIkSW1hzZqlXH317hW1iRO/yMSJny6oI20pA60kSZKkltbdvZY5cyrnw26//Qkce+xVBXWkgWKglSRJktSyaq1cPH16NxHVdTUfA60kSZKklnPFFduxfn1XRe2kk55kxIgdCupIg8FAK0mSJKll3HPPR3nooW9V1I49di7bb//iYhrSoDLQSpIkSWp6TzzxB26++RUVtX33PZcJEz5VUEcaCgZaSZIkSU2r1srFo0YdwnHH3V5QRxpKBlpJkiRJTSezm9mzh1fVOzqygG5UFAOtJEmSpKZSe+Xi9UQMK6AbFclAK0mSJKkpXHfd4axadVtFberUJWy99e59vEOtzkArSZIkqaE98MB53H//JytqRxzxO3be+RV9vEPtwkArSZIkqSE9/fR13HBD5eN29tzzgxx44L8U1JEa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\n",
      "text/plain": [
       "<Figure size 1152x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.rcParams['figure.figsize'] = [16, 9]\n",
    "plt.scatter(X_test, y_test,color='red')\n",
    "plt.plot(X, line,color='y');\n",
    "\n",
    "plt.title('Testing Set',fontdict=font1)\n",
    "plt.xlabel('Hours studied',fontdict=font2)\n",
    "plt.ylabel('Percentage Score',fontdict=font2)\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59eb38d0",
   "metadata": {},
   "source": [
    "# Testing in Hours and Predicting Scores "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "9373914f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.5]\n",
      " [3.2]\n",
      " [7.4]\n",
      " [2.5]\n",
      " [5.9]]\n"
     ]
    }
   ],
   "source": [
    "print(X_test)  \n",
    "y_pred=linreg.predict(X_test)  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "01a0f536",
   "metadata": {},
   "source": [
    "# Comparing Actual vs Predicted Scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "3a80e8dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([20, 27, 69, 30, 62], dtype=int64)"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "d8bd8fd1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([16.88414476, 33.73226078, 75.357018  , 26.79480124, 60.49103328])"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7559a67a",
   "metadata": {},
   "source": [
    "# Comparison of Actual vs Predicted values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "56ea2993",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual value</th>\n",
       "      <th>Predicted value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>16.884145</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>27</td>\n",
       "      <td>33.732261</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>69</td>\n",
       "      <td>75.357018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>30</td>\n",
       "      <td>26.794801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62</td>\n",
       "      <td>60.491033</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Actual value  Predicted value\n",
       "0            20        16.884145\n",
       "1            27        33.732261\n",
       "2            69        75.357018\n",
       "3            30        26.794801\n",
       "4            62        60.491033"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "comp=pd.DataFrame({'Actual value': y_test,\"Predicted value\": y_pred})\n",
    "compss"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb7f0930",
   "metadata": {},
   "source": [
    "# Testing with the provided Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "1711cb1a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The No.of hours= 9.25 hrs ,Predicted Score= 93.69173248737538\n"
     ]
    }
   ],
   "source": [
    "hours=9.25\n",
    "own_pred=linreg.predict([[hours]])\n",
    "print('The No.of hours=',hours, 'hrs',',Predicted Score=',own_pred[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b9337c2",
   "metadata": {},
   "source": [
    "# Evaluating the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "b1b0efd8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error= 4.183859899002975\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import mean_absolute_error\n",
    "err=mean_absolute_error(y_test,y_pred)\n",
    "print('Mean Absolute Error=',err)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
