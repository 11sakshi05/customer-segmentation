from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
kmeans = pickle.load(open("model/kmeans_model.pkl", "rb")) #same as we taken in .ipynb file to save model
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# Cluster meaning dictionary
segment_meaning = {
    0: {
        "title": "Average Customers 🙂",#moderate income and moderate spending
        "desc": "Customers with moderate income and balanced spending habits.",
        "recommend": "Target with loyalty programs, combo offers, and seasonal promotions."
    },
    1: {
        "title": "Premium Customers 💎",#high income and high spending 
        "desc": "High-income customers with high spending behavior.",
        "recommend": "Offer VIP membership, exclusive products, and premium services."
    },
    2: {
        "title": "Impulsive Buyers 🎉", #low income and high spending
        "desc": "Customers with lower income but high spending score.",
        "recommend": "Use flash sales, limited-time discounts, and social media promotions."
    },
    3: {
        "title": "Careful High-Income Customers 💰",#high income and low spending 
        "desc": "High earners but cautious spenders.",
        "recommend": "Provide value deals, detailed product benefits, and long-term savings offers."
    },
    4: {
        "title": "Budget Customers 🧍", #low income and low spending 
        "desc": "Low-income customers with low spending patterns.",
        "recommend": "Attract with discounts, coupons, and affordable product ranges."
    }
}



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    income = float(request.form['income'])
    spending = float(request.form['spending'])

    # Scale input
    sample = np.array([[income, spending]])
    sample_scaled = scaler.transform(sample)  #pass input in scaler function inside a trained model

    # Predict cluster
    cluster = kmeans.predict(sample_scaled)[0]  #pass scaled valued in cluster trained in model
    #print("Input:", income, spending)
    #print("Scaled:", sample_scaled)
    #print("Predicted cluster:", cluster)

    result = segment_meaning[cluster]  #it will return the output from the model after applying k means cluster 

    return render_template("index.html", 
                           prediction_title=result["title"],
                            prediction_desc=result["desc"],
                             prediction_recommend=result["recommend"])


if __name__ == "__main__":   #helps to run the flask app
    app.run(debug=True)
