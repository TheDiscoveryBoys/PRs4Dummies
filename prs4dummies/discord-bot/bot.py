"""
Discord Bot for PRs4Dummies

This bot listens for mentions in Discord channels and communicates with the PR analysis API
to provide AI-powered answers about pull requests.
"""

import os
import asyncio
import logging
import time
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
BOT_NAME = os.getenv('BOT_NAME', 'PR-Analyst')

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
    help_command=None  # We'll create a custom help command
)

class PRs4DummiesBot:
    """Main bot class for handling PR analysis requests."""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.bot_name = BOT_NAME
        self.thinking_messages = {}  # Track thinking messages per channel
        
    async def send_thinking_message(self, channel):
        """Send a 'thinking' message to indicate processing."""
        thinking_msg = await channel.send("🤔 **Thinking...** Analyzing your question about pull requests...")
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
                title="🤖 PR Analysis Result",
                description=data.get("answer", "No answer received"),
                color=0x00ff88,  # Green color
                timestamp=discord.utils.utcnow()
            )
            
            # Add question field
            embed.add_field(
                name="❓ Question",
                value=data.get("question", "Unknown"),
                inline=False
            )
            
            # Add processing time
            if "processing_time_ms" in data:
                embed.add_field(
                    name="⏱️ Processing Time",
                    value=f"{data['processing_time_ms']}ms",
                    inline=True
                )
            
            # Add sources information
            if "sources" in data and data["sources"]:
                sources_text = f"📚 {len(data['sources'])} sources used"
                if "total_sources" in data:
                    sources_text += f" (from {data['total_sources']} available)"
                embed.add_field(
                    name="📖 Sources",
                    value=sources_text,
                    inline=True
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
                title="🤖 PR-Analyst Bot",
                description="I'm here to help you understand pull requests! Mention me with a question like:",
                color=0x0099ff
            )
            help_embed.add_field(
                name="Example Questions",
                value="• @PR-Analyst what is this PR about?\n• @PR-Analyst explain the changes in this PR\n• @PR-Analyst what are the main features added?",
                inline=False
            )
            help_embed.add_field(
                name="Commands",
                value=f"• `{BOT_PREFIX}help` - Show this help\n• `{BOT_PREFIX}status` - Check API status\n• `{BOT_PREFIX}info` - Get system information",
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
        title="🤖 PR-Analyst Bot Help",
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
        value="• @PR-Analyst what is this PR about?\n• @PR-Analyst explain the changes in this PR\n• @PR-Analyst what are the main features added?\n• @PR-Analyst what issues does this PR fix?",
        inline=False
    )
    
    help_embed.add_field(
        name="Commands",
        value=f"• `{BOT_PREFIX}help` - Show this help\n• `{BOT_PREFIX}status` - Check API status\n• `{BOT_PREFIX}info` - Get system information",
        inline=False
    )
    
    help_embed.add_field(
        name="About",
        value="This bot uses advanced AI to analyze pull request data and provide intelligent answers based on the repository content.",
        inline=False
    )
    
    await ctx.send(embed=help_embed)

@bot.command(name="status")
async def status_command(ctx):
    """Check the status of the PR analysis API."""
    try:
        url = f"{pr_bot.api_base_url}/health"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            status_embed = discord.Embed(
                title="✅ API Status",
                description="The PR analysis API is running and healthy!",
                color=0x00ff88,
                timestamp=discord.utils.utcnow()
            )
            
            # Add status details
            if "vector_store_info" in data:
                vs_info = data["vector_store_info"]
                status_embed.add_field(
                    name="📊 Vector Store",
                    value=f"Status: {vs_info.get('status', 'Unknown')}\nDocuments: {vs_info.get('document_count', 'Unknown')}",
                    inline=True
                )
            
            if "uptime_seconds" in data:
                uptime = int(data["uptime_seconds"])
                hours = uptime // 3600
                minutes = (uptime % 3600) // 60
                status_embed.add_field(
                    name="⏰ Uptime",
                    value=f"{hours}h {minutes}m",
                    inline=True
                )
            
            await ctx.send(embed=status_embed)
        else:
            error_embed = discord.Embed(
                title="❌ API Status",
                description="The PR analysis API is not responding properly.",
                color=0xff8800,
                timestamp=discord.utils.utcnow()
            )
            await ctx.send(embed=error_embed)
            
    except Exception as e:
        error_embed = discord.Embed(
            title="❌ API Status",
            description="Could not connect to the PR analysis API.",
            color=0xff4444,
            timestamp=discord.utils.utcnow()
        )
        error_embed.add_field(
            name="Error",
            value=str(e),
            inline=False
        )
        await ctx.send(embed=error_embed)

@bot.command(name="info")
async def info_command(ctx):
    """Get information about the PR analysis system."""
    try:
        url = f"{pr_bot.api_base_url}/info"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            info_embed = discord.Embed(
                title="ℹ️ System Information",
                description="Information about the PR analysis system",
                color=0x0099ff,
                timestamp=discord.utils.utcnow()
            )
            
            # Add system details
            if "vector_store" in data:
                vs_info = data["vector_store"]
                info_embed.add_field(
                    name="📊 Vector Store",
                    value=f"Status: {vs_info.get('status', 'Unknown')}\nDocuments: {vs_info.get('document_count', 'Unknown')}\nModel: {vs_info.get('embedding_model', 'Unknown')}",
                    inline=False
                )
            
            if "embedding_model" in data:
                info_embed.add_field(
                    name="🧠 AI Models",
                    value=f"Embedding: {data['embedding_model']}\nLLM: {data.get('llm_type', 'Unknown')}",
                    inline=True
                )
            
            if "api_version" in data:
                info_embed.add_field(
                    name="🔧 API",
                    value=f"Version: {data['api_version']}",
                    inline=True
                )
            
            await ctx.send(embed=info_embed)
        else:
            await ctx.send("❌ Could not retrieve system information.")
            
    except Exception as e:
        await ctx.send(f"❌ Error getting system information: {str(e)}")

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
