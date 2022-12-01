import requests
from datetime import datetime

k = 0

def call(k):
    res = requests.post("http://localhost:8080/derivedNLFeatures?documentType=REVIEWS&batch-size=10&from=" + str(k))
    if not isinstance(res.content, int) or res.content != -1:
        current_time = datetime.now().strftime("%H:%M:%S")
        print("Recovered once at " + str(current_time))
        req2 = requests.get("http://localhost:8080/getLastReview")
        x = int(req2.content) - 10
        print("Starting again at " + str(x))
        call(x)

call(k)