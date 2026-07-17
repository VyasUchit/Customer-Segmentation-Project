Problem Statement :

What problem am I solving?

Businesses often have a large number of customers with different purchasing habits, preferences, 
and spending behaviors. Treating every customer the same makes marketing campaigns less effective
 and can lead to lower customer satisfaction and reduced revenue.

This project aims to identify groups of customers with similar characteristics using unsupervised machine learning.
 By discovering these hidden customer segments, businesses can better understand their customers and make informed decisions 
 for marketing, sales, and customer engagement.

Who will use this application?

This application can be used by:

Marketing Teams to create targeted marketing campaigns.
Sales Teams to identify high-value customers and improve sales strategies.
Business Analysts to analyze customer behavior and discover trends.
Company Management to make data-driven business decisions.
Retail and E-commerce Businesses to improve customer retention and increase revenue.
Why is customer segmentation important?

Every customer is different. Some customers purchase frequently, some spend more money, while others may only shop during
discounts.

Customer segmentation helps businesses:

Understand different customer groups.
Identify high-value and loyal customers.
Improve customer satisfaction through personalized experiences.
Optimize marketing campaigns by targeting the right audience.
Detect customers who are at risk of becoming inactive.
Allocate resources more efficiently.

Instead of using a single marketing strategy for everyone, businesses can design specific strategies for each customer segment.

What business value does it provide?

Implementing customer segmentation provides several business benefits:

Increased sales through personalized marketing campaigns.
Better customer retention by identifying valuable customers.
Improved return on marketing investment (ROI).
Enhanced customer experience through tailored recommendations.
More efficient resource allocation.
Data-driven decision-making based on customer behavior.
Identification of new business opportunities and growth strategies.

Overall, customer segmentation enables businesses to understand their customers more effectively and make strategic
 decisions that improve both customer satisfaction and business profitability.


Dataset Information :

People

ID: Customer's unique identifier
Year_Birth: Customer's birth year
Education: Customer's education level
Marital_Status: Customer's marital status
Income: Customer's yearly household income
Kidhome: Number of children in customer's household
Teenhome: Number of teenagers in customer's household
Dt_Customer: Date of customer's enrollment with the company
Recency: Number of days since customer's last purchase
Complain: 1 if the customer complained in the last 2 years, 0 otherwise

Products

MntWines: Amount spent on wine in last 2 years
MntFruits: Amount spent on fruits in last 2 years
MntMeatProducts: Amount spent on meat in last 2 years
MntFishProducts: Amount spent on fish in last 2 years
MntSweetProducts: Amount spent on sweets in last 2 years
MntGoldProds: Amount spent on gold in last 2 years

Promotion

NumDealsPurchases: Number of purchases made with a discount
AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
Place

NumWebPurchases: Number of purchases made through the company’s website
NumCatalogPurchases: Number of purchases made using a catalogue
NumStorePurchases: Number of purchases made directly in stores
NumWebVisitsMonth: Number of visits to company’s website in the last month
Target
Need to perform clustering to summarize customer segments.

Education Column : 

| Education      | Meaning                                                                                                                |
| -------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Basic**      | Primary or basic school education (up to around 8th/10th grade, depending on the country)                              |
| **2n Cycle**   | **Second Cycle Education**, which usually refers to **high school or secondary education** (below a bachelor's degree) |
| **Graduation** | Bachelor's degree                                                                                                      |
| **Master**     | Master's degree                                                                                                        |
| **PhD**        | Doctoral degree                                                                                                        |

Marital_Status :

| Marital Status | Meaning                                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Married**    | Customer is legally married.                                                                                                   |
| **Together**   | Customer is living with a partner but may not be legally married (similar to a live-in relationship or long-term partnership). |
| **Single**     | Customer is unmarried.                                                                                                         |
| **Divorced**   | Customer was married but is now divorced.                                                                                      |
| **Widow**      | Customer's spouse has passed away.                                                                                             |
| **Alone**      | Customer lives alone. This is very similar to "Single" and may represent the same type of customer.                            |
| **Absurd**     | An unusual category that doesn't represent a real marital status. Most likely a data entry error or placeholder.               |
| **YOLO**       | "You Only Live Once." This is not a valid marital status and is almost certainly an incorrect or humorous entry.               |


Cluster 2
------------
High income
High spending
Frequent buyers

Recommendation:
Offer premium membership
Give exclusive offers
Cross-sell luxury products