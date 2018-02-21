requestDict = {
      'instances': 
            [
                  {
                        "Lat": 37.750179, "Long": -122.421427, "Altitude": 33.0, "Date_": "7/5/17", "Time_": "23:37:25", "dt_": "7/4/17 23:37"
                  }
            ]     
}

import json
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
credentials = GoogleCredentials.get_application_default()
ml = discovery.build('ml','v1', credentials=credentials)

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors

# Store your full project ID in a variable in the format the API needs.
projectID = 'sandboai-184920'

# Get application default credentials (possible only if the gcloud tool is
#  configured on your machine).
credentials = GoogleCredentials.get_application_default()

# Build a representation of the Cloud ML API.
mlapi = discovery.build('ml', 'v1', credentials=credentials, discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')


# Create a request to call projects.models.create.
parent = 'projects/%s/models/%s/versions/%s' % (projectID, 'trajectory', 'v2')
request = mlapi.projects().predict(
              name=parent, body=requestDict).execute()

print('response{}'.format(request))

