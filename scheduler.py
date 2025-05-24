from apscheduler.schedulers.background import BackgroundScheduler
import time
import mysql.connector
from mysql.connector import Error
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

        
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            port='3306',
            database='trace_route',
            user='root',
            password=''
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


def print_hello():
    print("Hello")


def fetch_data():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT r.user_id, b.route_no, r.booktime
        FROM reservation r
        JOIN buses b ON r.bus_no = b.bus_no
        WHERE r.status = 'COMPLETED'
    """)
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return data


def preprocess_data(data):
    features = []
    target = []
    
    # Get all unique user IDs and route numbers for encoding
    user_ids = list(set(row['user_id'] for row in data))
    route_nos = list(set(row['route_no'] for row in data))

    le_user = LabelEncoder()
    le_user.fit(user_ids)  # Fit on all unique user_ids

    le_route = LabelEncoder()
    le_route.fit(route_nos)  # Fit on all unique route_nos

    for row in data:
        user_id = le_user.transform([row['user_id']])[0]  
        route_no = le_route.transform([row['route_no']])[0]
        
        booktime = row['booktime']
        hour_of_day = booktime.hour
        weekday = booktime.weekday()

        features.append([user_id, hour_of_day, weekday])
        target.append(route_no)
    
    return features, target, le_user, le_route


def train_model():
    data = fetch_data()
    if not data:
        print("No data found for training.")
        return

    features, target, le_user, le_route = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoders
    with open('route_predictor.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('user_encoder.pkl', 'wb') as f:
        pickle.dump(le_user, f)
    with open('route_encoder.pkl', 'wb') as f:
        pickle.dump(le_route, f)

    print("Model training complete and saved.")


def start_scheduler():
    scheduler = BackgroundScheduler()
    # scheduler.add_job(train_model, 'interval', seconds=5)  # Runs every 2 seconds
    scheduler.add_job(train_model, 'cron', hour=0, minute=0)  # Runs every day at 12:00 AM
    scheduler.start()

if __name__ == "__main__":
    start_scheduler()
    try:
        while True:
            time.sleep(1)  # Keeps the script running
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down scheduler...")
