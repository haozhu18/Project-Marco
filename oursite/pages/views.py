from django.shortcuts import render
import json

# Create your views here.
def home_view(request):
    '''
    This function handles the rendering of the home page

    Input: request from the browser

    Ouput: rendered webpage
    '''

    # clear json
    dictionary = {}
    with open('result.json', 'w') as fp:
        json.dump(dictionary, fp)
    my_context = {'title': 'Project Marco',
                  'msg': "life is short, but the world is big"}
    return render(request, 'home.html', my_context)

def team_view(request):
    '''
    This function handles the rendering of the team page

    Input: request from the browser

    Ouput: rendered webpage
    '''
    
    team_context = {'team': ['Abhimanyu Choudhary', 
                             'Ezra Max',
                             'Hao Zhu ',
                             'Shiyu Tian' 
                             ]}
    return render(request, 'team.html', team_context)