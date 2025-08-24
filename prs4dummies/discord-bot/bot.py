"""
Discord Bot for PRs4Dummies

This bot listens for mentions in Discord channels and communicates with the PR analysis API
to provide AI-powered answers about pull requests.
"""

import os
import logging
import random
from typing import Optional
import discord
from discord.ext import commands
import requests
import json

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
BOT_PREFIX = os.getenv('BOT_PREFIX', '!')
BOT_NAME = os.getenv('BOT_NAME', 'GitWit')

# Validate required environment variables
if not DISCORD_TOKEN:
    logger.error("DISCORD_TOKEN environment variable is required!")
    exit(1)

# Bot setup with intents
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=BOT_PREFIX,
    intents=intents,
    help_command=None
)

class PRs4DummiesBot:
    """Main bot class for handling PR analysis requests."""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.bot_name = BOT_NAME
        self.thinking_messages = [
            "⚙️ **Analyzing the diffs...** Just a moment while I review the changes.",
            "🧠 **Consulting the AI core...** Synthesizing an answer from the repo's knowledge base.",
            "💻 **Digging through the commits...** Searching for the insights you need.",
            "🤖 **GitWit is on the case!** Examining the pull request details now.",
            "🔍 **Scanning the codebase...** Looking for clues to answer your question."
        ]
        
    async def send_thinking_message(self, channel):
        """Send a random 'thinking' message to indicate processing."""
        message_text = random.choice(self.thinking_messages)
        thinking_msg = await channel.send(message_text)
        return thinking_msg
    
    async def delete_thinking_message(self, thinking_msg):
        """Delete the thinking message."""
        try:
            await thinking_msg.delete()
        except discord.NotFound:
            pass  # Message already deleted
        except Exception as e:
            logger.warning(f"Could not delete thinking message: {e}")
    
    def format_response(self, api_response):
        """Format the API response for Discord display."""
        try:
            # Parse the API response
            if isinstance(api_response, dict):
                data = api_response
            else:
                data = json.loads(api_response)
            
            # Create a formatted embed
            embed = discord.Embed(
                title="🤖 GitWit's Answer",
                description=data.get("answer", "No answer received"),
                color=0x00ff88,  # Green color
                timestamp=discord.utils.utcnow()
            )
            
            # Add question field
            embed.add_field(
                name="❔Your Question",
                value=data.get("question", "Unknown"),
                inline=False
            )
            
            # Add footer
            embed.set_footer(text="Powered by PRs4Dummies RAG API")
            
            return embed
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            # Fallback to simple text
            return f"**Answer:** {api_response.get('answer', 'Error processing response')}"
    
    async def ask_api(self, question: str) -> Optional[dict]:
        """Send a question to the PR analysis API."""
        try:
            url = f"{self.api_base_url}/ask"
            payload = {
                "question": question,
                "include_sources": True
            }
            
            logger.info(f"Sending question to API: {question}")
            
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to API")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling API: {e}")
            return None
    
    async def handle_question(self, message, question: str):
        """Handle a question from a user."""
        channel = message.channel
        
        # Send thinking message
        thinking_msg = await self.send_thinking_message(channel)
        
        try:
            # Get answer from API
            api_response = await self.ask_api(question)
            
            if api_response:
                # Format and send response
                formatted_response = self.format_response(api_response)
                await channel.send(embed=formatted_response)
            else:
                # Send error message
                error_embed = discord.Embed(
                    title="❌ Error",
                    description="Sorry, I couldn't get an answer from the PR analysis system. Please try again later.",
                    color=0xff4444,  # Red color
                    timestamp=discord.utils.utcnow()
                )
                await channel.send(embed=error_embed)
                
        except Exception as e:
            logger.error(f"Error handling question: {e}")
            error_embed = discord.Embed(
                title="❌ Error",
                description="An unexpected error occurred while processing your question.",
                color=0xff4444,
                timestamp=discord.utils.utcnow()
            )
            await channel.send(embed=error_embed)
        
        finally:
            # Clean up thinking message
            await self.delete_thinking_message(thinking_msg)

# Initialize the bot handler
pr_bot = PRs4DummiesBot()

@bot.event
async def on_ready():
    """Called when the bot is ready."""
    logger.info(f"Bot is ready! Logged in as {bot.user}")
    logger.info(f"Bot ID: {bot.user.id}")
    logger.info(f"Connected to {len(bot.guilds)} guilds")
    
    # Set bot status
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="for PR questions"
        )
    )

@bot.event
async def on_message(message):
    """Handle incoming messages."""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Check if bot was mentioned
    if bot.user.mentioned_in(message):
        # Extract the question (remove the mention)
        question = message.content.replace(f"<@{bot.user.id}>", "").strip()
        question = question.replace(f"<@!{bot.user.id}>", "").strip()
        
        # Remove common prefixes
        for prefix in [f"@{bot.user.name}", f"@{BOT_NAME}", "hey", "hi", "hello"]:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()
        
        # Check if there's actually a question
        if not question:
            help_embed = discord.Embed(
                title="🤖 GitWit Bot",
                description="I'm here to help you understand pull requests! Mention me with a question like:",
                color=0x0099ff
            )
            help_embed.add_field(
                name="Example Questions",
                value="• @GitWit what is PR #5 about?\n• @GitWit explain the changes in PR #23\n• @GitWit what are the main features added in PR #11?",
                inline=False
            )
            help_embed.add_field(
                name="Commands",
                value=f"• `{BOT_PREFIX}help` - Show this help",
                inline=False
            )
            await message.channel.send(embed=help_embed)
            return
        
        # Process the question
        await pr_bot.handle_question(message, question)
    
    # Process commands
    await bot.process_commands(message)

@bot.command(name="help")
async def help_command(ctx):
    """Show help information."""
    help_embed = discord.Embed(
        title="🤖 GitWit Bot Help",
        description="I'm an AI-powered bot that helps you understand pull requests using the PRs4Dummies system.",
        color=0x0099ff
    )
    
    help_embed.add_field(
        name="How to Use",
        value="Simply mention me with a question about pull requests!",
        inline=False
    )
    
    help_embed.add_field(
        name="Example Questions",
        value="• @GitWit what is PR #5 about?\n• @GitWit explain the changes in PR #23\n• @GitWit what are the main features added in PR #11?",
        inline=False
    )
    
    help_embed.add_field(
        name="Commands",
        value=f"• `{BOT_PREFIX}help` - Show this help",
        inline=False
    )
    
    help_embed.add_field(
        name="About",
        value="This bot uses advanced AI to analyze pull request data and provide intelligent answers based on the repository content.",
        inline=False
    )
    
    await ctx.send(embed=help_embed)


@bot.event
async def on_command_error(ctx, error):
    """Handle command errors."""
    if isinstance(error, commands.CommandNotFound):
        return  # Ignore unknown commands
    
    error_embed = discord.Embed(
        title="❌ Command Error",
        description=f"An error occurred while processing the command: {str(error)}",
        color=0xff4444,
        timestamp=discord.utils.utcnow()
    )
    
    await ctx.send(embed=error_embed)

def main():
    """Main function to run the bot."""
    logger.info("Starting PRs4Dummies Discord Bot...")
    logger.info(f"API Base URL: {API_BASE_URL}")
    logger.info(f"Bot Name: {BOT_NAME}")
    
    try:
        bot.run(DISCORD_TOKEN)
    except discord.LoginFailure:
        logger.error("Invalid Discord token!")
        exit(1)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        exit(1)

if __name__ == "__main__":
    main()
