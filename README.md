# Does Elon Musk tweets affect Tesla Stock Price? What is tesla stock price with and without his tweets?
## Steps:
1. Collecting data
2. EDA
3. Time Series (TSLA); Output: Residuals
4. NLP Modeling (Tweets)
5. Joining the results: Boosting TSLA Predicton

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/fb03336b-8867-4198-851e-dbf383b8124f)

### Step 1: Collecting Data
For this project, two dataset were required: 
- Tesla Stock Price over time
- Elon Musk Tweets

For stock price, I used yahoo finance to collect data by python.

For Elon Musk Tweets, I downloaded a dataset from Kaggle as it is no longer free to use Tweeter API.

### Step 2: Exploratory data analysis
Here I have included some of the EDA. 

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/cd19d9a6-ac02-48f7-a039-d5c48dd32433)
![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/f3a78e5b-b22b-4b12-9513-da6bae6bad53)

#### EDA on Tweets

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/a46a0b7d-c98a-426a-99a5-588f5730b80b)

#### Tweets Filtering Process

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/b2dd10b0-394f-4e81-a7f7-fffd52ded9d0)

####  EDA on combining sample Tweets and Tesla Stock Price

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/b2269544-3e6b-4989-b6c4-ba5c8de5708d)
![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/d18eaea8-8e35-41b7-b8b4-98f9c678cb67)

### Step 3: Modeling

- TSLA Price Prediction 
- Elon’s Musk tweet’s affect prediction

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/e3651838-5c18-4ef9-b19c-2723c9f599c0)

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/7a170362-d762-426b-a695-6f9431db416b)

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/73b1e9b9-06f7-44b1-aed7-ca0c63b37f16)

#### Residual sign as NLP Modeling Target

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/ef7b1e72-0ef9-4e4b-a9f6-0395cb8ebc39)

## Step 4: NLP Classification Models to predict tweets affect on residuals

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/1ab77934-b5b4-4e52-999a-fb65c3911dc9)

![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/7ffa00e9-8d06-4a41-9f27-5a47c501542a)
## Step 5: Boosting TSLA Prediction!
#### Combining the results in one table
![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/e31a49bf-5ac8-4074-9f8c-548aa5db0973)
#### Compare boosted model with initial time series winner!
![image](https://github.com/taaaraaa/tsla-price-prediction/assets/26361973/7f57df53-bc0f-4d05-9015-0d1edd7116eb)

## Future work possibility
- Adding computer vision models to include the images/videos for tweets interpretation!
- Including the tweets posted in languages other than English

