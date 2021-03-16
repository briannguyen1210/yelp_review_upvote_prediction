Yelp Review Upvote Prediction project is about finding the features from the data from Yelp to
determine the number of upvotes for a review. The data consists of features for the business,
users, check-ins, and reviews. For each group of data, we can access to the features such as
review texts, the star ratings, business information, etc. Therefore, we can use these features to
predict what will contribute the most to the review upvote.

After looking into all the features into consideration, I narrowed down 7 features that could
potentially affect the review upvotes. Those are:
1. Star Ratings
2. Review Length
3. Number of Key Words (e.g: awesome, great, nice, love, awful, etc)
4. Average Votes per Review by Users
5. Number of Reviews on a Business
6. Is the Business Open?
7. Number of Yelp Check-in of a Business

A Neural-Network model is used as a model training. After testing on each combination (of 2, 3, 4, or 5 out of 7), the 3 features that are likely
to be important for the number of upvotes for a review are StarRating, ReviewLength, and UserAvgVotePerReview with the model accuracy of 57.4%.

More info about the project and dataset is in the following link: 
https://www.kaggle.com/c/yelp-recruiting/data
