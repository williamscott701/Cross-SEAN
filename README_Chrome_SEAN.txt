Chrome-SEAN

Note
- Chrome-SEAN is not yet made public to protect the anonymity.
- All tweets detected and reported are logged and stored on server for further use.



Features
1. Detection of Twitter page (works only if the current page in view has tweet)
2. Detection of tweet ID on the client side. (By verifying the tweet ID on the client side, we reduce the load on the server)
3. Retrieving the sentiment and tweet text as soon as step 2 detects the tweet ID
4. Ask Cross-SEAN gives the result of Fake/Non-Fake with the confidence score.
5. Users can report whether the tweet is Genuine or Fake, irrespective of the class detected.



Robustness
- The Flask application is run on the server, connecting to Redis (handling multiple incoming connections)
- Load balancing and DDoS attacks are handled by using Redis and Beanstalkd.



Live Demo of extension
- After installation of the extension, train Cross-SEAN referring to README_Cross_SEAN.txt.
- Run Flask:
  - cd Cross-SEAN
  - export FLASK_APP=flask_sean
  - flask run --host 0.0.0.0
  - change the IP address and Port from http://192.168.29.83:5000 to your_custom_ip_port in script.js in Chrome_SEAN.zip
- Follow the instructions to install the extension.



How to install extension?
- Go to chrome://extensions in your google chrome browser.
- Switch to developer mode. 
- Click on the Load Unpacked option and upload the ‘Chrome-SEAN’ folder which you can extract from the 'Chrome-Sean.zip' present as a sub-folder in this directory.
- You should be able to see the Chrome-SEAN extension in the toolbar.
- After following these steps one should be able to see and reproduce results as shown in Demo.mp4
