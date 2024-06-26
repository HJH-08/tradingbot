import quandl

quandl.ApiConfig.api_key = "g-cGHeLRESQ5Qx8gBUkD"
data = quandl.get('WIKI/FB')
print(data)