import requests
import json

url = 'https://www.transitlink.com.sg/eservice/eguide/service_route.php?service=147'
resp = requests.get(url)
# data = json.loads(resp.text)

print(resp.text)
