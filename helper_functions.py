import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity




# Get all the active products from the mongo database and convert it into dataframe
def getAllProductsDataframe(mongo):
    pipeline = [
        {
            "$match": {
                "status": 1,
            }
        },
        {
            "$sort": {
                "createdAt": -1
            }
        },
        {
            "$addFields": {
                "product_id": { "$toString": "$_id" },
                "category": { "$toString": "$category" },
                "sub_category": { "$toString": "$sub_category" },
                "model": { "$toString": "$model" }
            }
        },
        {
            "$project": {
                "product_id": 1,
                "name": 1,
                "price": 1,
                "rating": 1,
                "category": 1,
                "sub_category": 1,
                "model": 1,
                "brand": 1,
                "_id": 0  # Exclude the original _id field
            }
        }
    ]
    try:
        products = list(mongo.db.products.aggregate(pipeline))
    except Exception as e:
        print("ERROR", e)

    # Creating a dataframe
    product_dataframe= pd.DataFrame(products)
    product_dataframe.set_index('product_id', inplace=True)
    print("\n------------------------------Products DataFrame has been created.------------------------------\n")
    return product_dataframe


# Get all the active orders from the mongo database and convert it into dataframe
def getAllOrdersDataframe(mongo):
    pipeline = [
        {
            "$match": {
                "status": 1
            }
        },
        {
            "$sort": {
                "createdAt": -1
            }
        },
        {
            "$unwind": "$orderItems"
        },
        {
            "$lookup": {
                "from": "products",
                "localField": "orderItems.product_id",
                "foreignField": "_id",
                "as": "product_info"
            }
        },
        {
            "$unwind": {
                "path": "$product_info",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$addFields": {
                "orderItems.product_id": {
                    "_id": { "$toString": "$product_info._id" },
                    "name": "$product_info.name",
                    "price": "$product_info.price",
                    "rating": "$product_info.rating",
                    "category": { "$toString": "$product_info.category" },
                    "sub_category": { "$toString": "$product_info.sub_category" }
                },
                "orderItems.name": "$product_info.name",
                "orderItems.price": "$product_info.price",
                "orderItems.image": {
                    "$cond": {
                        "if": { "$isArray": "$product_info.variations" },
                        "then": {
                            "$map": {
                                "input": "$product_info.variations",
                                "as": "variation",
                                "in": "$$variation.images"
                            }
                        },
                        "else": []
                    }
                }
            }
        },
        {
            "$group": {
                "_id": { "$toString": "$_id" },
                "shippingInfo": { "$first": "$shippingInfo" },
                "userId": { "$first": "$userId" },
                "payment_method": { "$first": "$payment_method" },
                "itemsPrice": { "$first": "$itemsPrice" },
                "taxPrice": { "$first": "$taxPrice" },
                "shippingPrice": { "$first": "$shippingPrice" },
                "totalPrice": { "$first": "$totalPrice" },
                "orderStatus": { "$first": "$orderStatus" },
                "status": { "$first": "$status" },
                "orderItems": { "$push": "$orderItems" }
            }
        },
        {
            "$project": {
                "shippingInfo": 1,
                "userId": 1,
                "payment_method": 1,
                "itemsPrice": 1,
                "taxPrice": 1,
                "shippingPrice": 1,
                "totalPrice": 1,
                "orderStatus": 1,
                "status": 1,
                "orderItems.name": 1,
                "orderItems.price": 1,
                "orderItems.quantity": 1,
                "orderItems.product_id": 1,
                "orderItems.colour": 1,
                "orderItems.size": 1,
                "orderItems.dealer": 1
            }
        }
    ]
    orders = list(mongo.db.orders.aggregate(pipeline))

    records = []
    for order in orders:
        for item in order['orderItems']:
            records.append({
                'user_id': order['userId'],
                'product_id': item['product_id']['_id'],
                'rating': item['product_id']['rating'],
                'product_name': item['name'],
                'category': item['product_id']['category'],
                'sub_category': item['product_id']['sub_category'],
                'quantity': item['quantity']
            })
    order_dataframe = pd.DataFrame(records)
    print("\n------------------------------Orders DataFrame has been created.------------------------------\n")
    return order_dataframe



def createMatrixTables(order_dataframe):
    # Create interaction matrix
    interaction_matrix = pd.pivot_table(order_dataframe, index='user_id', columns='product_id', values='rating', fill_value=0)
    print("\n------------------------------Interaction Matrix has been created.------------------------------\n")

    # Calculate item-item similarity
    item_similarity_matrix = cosine_similarity(interaction_matrix.T)  # Transpose the interaction matrix
    item_similarity_df = pd.DataFrame(item_similarity_matrix, index=interaction_matrix.columns, columns=interaction_matrix.columns)
    print("\n------------------------------Item Similarity Dataframe has been created.------------------------------\n")

    return interaction_matrix, item_similarity_df




def recommend_products(user_id, interaction_matrix, item_similarity_df, product_dataframe, top_n=10):
    if user_id in interaction_matrix.index:
        # For existing users, generate personalized recommendations
        user_purchases = interaction_matrix.loc[user_id]
        # Calculate similarity-weighted purchase history
        similar_scores = item_similarity_df.dot(user_purchases).sort_values(ascending=False)
        # Filter out items already purchased by the user
        recommendations = similar_scores[~user_purchases.index.isin(user_purchases[user_purchases > 0].index)]
    else:
        # For new users, recommend top ordered products
        product_popularity = interaction_matrix.sum(axis=0).sort_values(ascending=False)
        recommendations = product_popularity.head(top_n)

    # Get top n recommendations and retrieve product details
    recommendations = recommendations.head(top_n).index
    recommendation_details = product_dataframe.loc[recommendations][['name']].reset_index()
    
    # Format recommendations as an array of objects with product id as key and name as value
    recommendations_array = [{"id": row['product_id'], "name": row['name']} for _, row in recommendation_details.iterrows()]
    # recommendations_array = [{product_id: product_dataframe.loc[product_id]['name']} for product_id in recommendation_details]
    
    return recommendations_array