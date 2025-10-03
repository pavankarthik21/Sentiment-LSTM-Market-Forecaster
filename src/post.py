import requests 

res=requests.post('https://httpsbin.org/post',json={"key":"value"})

print(res.text)