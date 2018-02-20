import api_builder as api

trajectory_data = {'instances':
  [
      {
        'dep_delay': 16.0,
        'taxiout': 13.0,
        'distance': 160.0,
        'avg_dep_delay': 13.34,
        'avg_arr_delay': 67.0,
        'carrier': 'AS',
        'dep_lat': 61.17,
        'dep_lon': -150.00,
        'arr_lat': 60.49,
        'arr_lon': -145.48,
        'origin': 'ANC',
        'dest': 'CDV'
      }
  ]
}

PROJECT = 'sandboai-184920'
parent = 'projects/%s/models/%s/versions/%s' % (PROJECT, 'trajectory', 'v2')
response = api.projects().predict(body=trajectory_data, name=parent).execute()
print("response={0}".format(response))