# AWS Lex & Bedrock

# AWS Lex
AWS Chatbot service, allows Voice and Text
Benifits: Simple, seamless deployment and scaling
ℹ️ Seamless Deployment: Deployment without disruption
### Components:
- **Bot**: Chatbot container
- **Intent**: Final Action or Goal User wants. Can be more than 1
 e.g Hotel Booking is the final intent (not rooms, nights questions)
- **fallback Intent**: If lex can't match with a specific one then goes to fallback
e.g Sorry I didn't understand. You can ask me to do A or B.
(It can )
- **Utterance**:  User input or phrases user might say to trigger intent
- **slots**: Variables bots need to complete task
- **Response Cards**: 

## Creating a Lex Bot


### Connecting Lambda to Lex
 