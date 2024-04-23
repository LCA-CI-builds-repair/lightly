# How This RThis README.md file contains information about running tests against the live API.ME file provides information on testing scenarios in the project:

1. Testing the `api_workflow_client` directly allows full configuration of its mocked version. This is useful for testing scenarios where the `api_workflow_client` is used directly.

2. Within the `tests/cli` directory, testing of the CLI commands takes place. In this context, a new `api_workflow_client` is created for each CLI command test. However, it does not support configuring the mocked `api_workflow_client`.ite tests for the API

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