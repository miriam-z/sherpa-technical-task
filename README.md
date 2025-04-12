## Context

At Sherpa, we develop AI-enabled software applications for management consulting and professional services firms, frequently handling complex, unstructured data sources such as:

- Excel spreadsheets (Survey data, business plans, etc.)
- PDF reports
- PowerPoint slide decks
- Word documents

Our solutions must be multi-tenant, highly secure, and optimised for enterprise scalability and reliability.

## 📌 Objective

You’ve been provided with a set of reports, articles and presentations on the topic of artificial intelligence, sourced from top-tier consulting firms (McKinsey, Bain and BCG). The documents include a mix of data — text, tables, charts, and diagrams. Your task is to build a prototype application, within 48 hours, that allows users to explore, synthesise and interrogate the content of these reports using AI-powered techniques.

- Python is preferred, but we welcome other languages if they’re your strength. Feel free to use any packages you wish
- Likewise, we can provide you with AzureOpenAI credentials, but you are also welcome to use an LLM of your choice.

The initial repository and dataset can be found in the below repo, which you create a fork of.

https://github.com/Charter-AI/sherpa-technical-task

> **Note:** We do not expect a fully-featured, enterprise grade solution. We’re evaluating your approach to problem-solving, handling ambiguity, and creating robust foundations. The task is intentionally open-ended to allow you to also show off your skills. You can choose which step of the process you’d like to focus on, based on your strengths and interests.
> 

## Deliverables

### Must-Haves

- A working codebase / repository that offers the user some ability to interact with the data. This can be as simple as the command line if you want to focus more heavily on the backend logic, but it could also be a fancy, deployed, UI if you want to show off your end-to-end development skills.
    - Please invite the following user to your repository: https://github.com/OLT2000
- README file documenting your thought process and set up instructions

### Nice-to-Haves

Aside from the basic chatbot set-up, we also place positive weightings on submissions that focus on some of the below concepts, as these are challenges that you will face on the job.

1. **Complex Data Parsing**
    1. Are you able to utilise all of the data contained in the reports (i.e. including the tables, charts and images which contain valuable information)?
    2. Typical consulting powerpoint decks contain content which does not follow a linear, logical flow like a Word document. Diagrams such as flow-charts use structure to add hierarchy to text.
2. **Authentication / RBAC**
    1. You may want to consider simulating user authentication, data siloes or user permissioning / access rights.
    2. Consulting data is often extremely confidential. Accidentally leaking data between companies, or even internal teams, could be detrimental to us — security must be paramount.
3. **Scalable Output Testing**
    1. What frameworks did you use to evaluate the quality of your model choices or prompts?
    2. As we explore different solutions to our data problems, we need a way to reliably compare model performance.
4. **Model output validation**
    1. How are you checking for hallucinations and response structures?
5. **Agentic Integration with APIs**
    1. Users may want to perform common desk research about an industry or company that may not be available in the existing datasets
6. **Scalability**
    1. Have you addressed the problem of scalability?
    2. Is your application set up to handle large volumes of concurrent requests? How are you ensuring systems experience no downtime