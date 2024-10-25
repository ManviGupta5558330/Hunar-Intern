import random
import re

class Chatbot:
    # Define responses for negative replies and exit commands
    negative_res = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later")

    def __init__(self):
        # Define regular expressions for different user intents
        self.alienbabble = {
            'describe_hunar_intern': r'.*\s*Hunar Intern.*',
            'about_courses': r'.*\s*courses.*',
            'about_internships': r'.*\s*internships.*',
            'skill_improvement_tips': r'.*\s*how to improve my skills.*',
            'course_recommendation': r'.*\s*recommend a course.*',
            'experience_with_hunar': r'.*\s*experience with hunar intern.*',
            'internship_interest': r'.*\s*internship.*',
            'how_are_you': r'.*\s*how are you.*',
            'what_are_you': r'.*\s*what are you.*',
            'who_made_you': r'.*\s*who made you.*',
            'who_is_creator': r'.*\s*who is your creator.*',
            'what_can_you_do': r'.*\s*what can you do.*'
        }

    def greet(self):
        # Ask for the user's name
        self.name = input("What is your name?\n")
        # Greet the user and explain the bot's purpose
        print(
            f"Hi {self.name}, I am manvi gupta's Chatbot. I can help you with your doubts related to Hunar Intern and career development.\n"
            "How can I help you today?\n"
        )
        # Start the chat
        self.chat()

    def make_exit(self, reply):
        # Check if the user's reply matches any exit commands
        for command in self.exit_commands:
            if reply == command:
                print("Have a nice day!\n")  # Goodbye message
                return True  # Return True to indicate the chat should end
        return False  # Continue if no exit command is matched

    def chat(self):
        # Prompt the user for questions
        reply = input("Feel free to ask your questions:\n").lower()
        while not self.make_exit(reply):  # Continue until an exit command is given
            reply = input(self.match_reply(reply))  # Get the response for the user's input

    def match_reply(self, reply):
        found_match = False  # Flag to track if a matching intent is found
        # Check the user's reply against known intents
        for intent, regex_pattern in self.alienbabble.items():
            if re.match(regex_pattern, reply):
                found_match = True  # Mark that a match was found
                # Call the appropriate response function based on the intent
                return getattr(self, intent)()  # Dynamically call the right method

        # If no matching intent was found, provide a default response
        if not found_match:
            return self.no_match_intent()

    def describe_hunar_intern(self):
        # Responses about Hunar Intern
        responses = (
            "Hunar Intern is a fantastic platform that provides internship opportunities specifically designed for enthusiastic students like you. "
            "It aims to bridge the gap between academic learning and practical industry experience.\n",
            "Hunar Intern not only offers internships but also provides quality courses across various computer science sub-branches, helping students prepare for their careers in technology.\n"
        )
        return random.choice(responses)  # Return a random response

    def about_courses(self):
        # Responses about available courses
        responses = (
            "We offer a wide range of courses, including web development, data science, machine learning, artificial intelligence, and more. "
            "Each course is tailored to provide practical knowledge and skills that are highly sought after in the industry.\n",
            "Our courses are designed to help you build a solid foundation and gain hands-on experience. You can choose based on your interests and career goals.\n"
        )
        return random.choice(responses)

    def about_internships(self):
        # Responses about internships
        responses = (
            "Internships are a critical part of your learning experience. At Hunar Intern, we connect you with companies looking for fresh talent, "
            "allowing you to apply your knowledge in real-world settings and gain invaluable experience.\n",
            "Through our internships, you can work on live projects, collaborate with experienced professionals, and enhance your resume significantly.\n"
        )
        return random.choice(responses)

    def skill_improvement_tips(self):
        # Responses providing skill improvement tips
        responses = (
            "To improve your skills, consider dedicating time to practice coding daily. Websites like LeetCode and HackerRank are great for this.\n"
            "Additionally, building personal projects can showcase your skills and help you learn new technologies.\n",
            "Joining online communities, attending workshops, or participating in hackathons can also provide great learning experiences.\n",
            "To enhance your resume, highlight your skills, relevant coursework, and any projects you've completed. Tailor your resume for each application to align with the job description.\n"
        )
        return random.choice(responses)

    def course_recommendation(self):
        # Responses recommending courses based on trends
        responses = (
            "Based on current trends, I recommend taking courses in data science and web development. These fields are rapidly growing and offer many job opportunities.\n",
            "If you're interested in artificial intelligence, our machine learning course could be a great fit for you. It's a cutting-edge field with lots of potential.\n"
        )
        return random.choice(responses)

    def experience_with_hunar(self):
        # Responses about experiences with Hunar Intern
        responses = (
            "Many students have reported positive experiences with Hunar Intern, noting the quality of courses and the helpfulness of mentors. "
            "It's a great place to gain both theoretical knowledge and practical skills.\n",
            "Hunar Intern is designed to support students in their career journeys. Many users appreciate the hands-on approach and the real-world applications of what they learn.\n"
        )
        return random.choice(responses)

    def internship_interest(self):
        # Response providing information about internships and a link
        response = (
            "It sounds like you're interested in internships! Hunar Intern offers a variety of internship opportunities that can help you gain practical experience.\n"
            "You can learn more about available internships and how to apply by visiting this link: https://www.hunarintern.live/learnmore\n"
        )
        return response

    def how_are_you(self):
        # Response to the question about how the bot is
        responses = (
            "I'm just a program, but I'm here and ready to assist you!\n",
            "As an AI, I don't have feelings, but I'm eager to help you with your questions!\n"
        )
        return random.choice(responses)

    def what_are_you(self):
        # Response to the question about what the bot is
        responses = (
            "I am manvi gupta's Chatbot, designed to assist you with questions related to Hunar Intern and career development.\n",
            "I'm a conversational AI created to provide information and guidance on internships and skill improvement.\n"
        )
        return random.choice(responses)

    def who_made_you(self):
        # Response to the question about who made the bot
        return "I was created by manvi gupta, a student at Sandip University in Nashik.\n"

    def who_is_creator(self):
        # Response to the question about the creator
        return ("My creator is manvi gupta, who designed me to assist you in your career journey.\n")

    def what_can_you_do(self):
        # Response explaining the bot's capabilities
        responses = (
            "I can answer your questions about Hunar Intern, available courses, internships, and tips for skill improvement.\n",
            "I am here to help you understand how to advance your career and make the most of your opportunities.\n"
        )
        return random.choice(responses)

    def no_match_intent(self):
        # Responses for unrecognized input
        responses = (
            "I'm not quite sure how to respond to that. Could you provide more details?\n",
            "That's interesting! Can you elaborate a bit more on that topic?\n",
            "I see. What do you mean by that? I'm eager to learn more!\n",
            "I'm here to assist you! What else would you like to know?\n"
        )
        return random.choice(responses)

# Instantiate the chatbot and start the greeting
bot = Chatbot()
bot.greet()