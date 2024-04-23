#There are three diffThis runs tests against the live API.

Additional context or information about running tests against the live API can be added here.nt locations to write tests for the API, each with its
own advantages and disadvantages:

1. In tests/api_workflow_client: 
    - Advantages: Allows testing the API workflow client directly.
    - Disadvantages: May not cover all API functionalities.

2. [Add Second Location Here]:
    - Advantages: [Add Advantages Here]
    - Disadvantages: [Add Disadvantages Here]

3. [Add Third Location Here]:
    - Advantages: [Add Advantages Here]
    - Disadvantages: [Add Disadvantages Here]to write tests for the API

There are three different locations to write tests for the API, each with its
own advantages and disadvantages:

1. In tests/api_workflow_client:
This is for testing the api_workflow_client directly with the ability to configure its mocked version fully.
Furthermore, you have a (partly) stateful api_workflow_client. 
2. In tests/cli:
This is for testing the cli commands. However, it will use a new api_workflow_client
for every new cli command. It does not allow configuring the mocked api_workflow_client.
3. In tests/UNMOCKED_end2end_tests: 
This runs tests against the live API. 