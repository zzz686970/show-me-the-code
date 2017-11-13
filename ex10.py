'''
    content: generate random numbers
    update: 2017-08-07
    author: Zhizhong Zhou
'''

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# sys.reload(sys)
driver = webdriver.PhantomJS(executable_path="/usr/local/Cellar/phantomjs/2.1.1/bin/phantomjs")
driver.get("http://mail.163.com/")
elem_user  = driver.find_element_by_name('username')
elem_user.send_keys("zzz686970@163.com")
elem_pwd = driver.find_element_by_name('password')
elem_pwd.send_keys("Zzh920708")
elem_pwd.send_keys(Keys.RETURN)
# data = driver.title
# # driver.save_screenshot("baidu.png")
# print (data)
time.sleep(5)
assert "百度" in driver.title
driver.close()
driver.quit()