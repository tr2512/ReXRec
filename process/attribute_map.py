def get_attribute_map(dataset):
    """Returns the dataset specific attribute names corresponding to 'user', 'item', 'rating', 'time' and 'review'."""
    return {
        "amazon" : {
            "user"   : "reviewerID",
            "item"   : "asin",
            "rating" : "overall",
            "time"   : "reviewTime",
            "review" : "reviewText"
        },
        "yelp" : {
            "user"   : "user_id",
            "item"   : "business_id",
            "rating" : "stars",
            "time"   : "date",
            "review" : "text"
        },
        "google" : {
            "user"   : "user_id",
            "item"   : "gmap_id",
            "rating" : "rating",
            "time"   : "time",
            "review" : "text"
        }
    }.get(dataset)