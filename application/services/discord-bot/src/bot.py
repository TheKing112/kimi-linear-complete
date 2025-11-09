# ✅ Hinzufügen am Anfang
import asyncio
from typing import Optional, Dict, Any  # ✅ Erweiterte Typ-Definitionen
import logging
import os
from datetime import datetime, timezone

# ✅ NEU - Für Rate Limiting
from collections import defaultdict
from time import time

import discord
from discord.ext import commands
import aiohttp
from fastapi import FastAPI
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from discord.ui import View, Button

# Import Remote Control
from remote_control import setup_remote_control

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kimi-bot-master")

# FastAPI Health App
health_app = FastAPI(
    title="Discord Bot Health API",
    version="2.1.0"
)

@health_app.get("/health")
async def health_check():
    """Bot Health Status"""
    uptime = (datetime.now(timezone.utc) - bot.start_time).total_seconds() if hasattr(bot, 'start_time') else 0
    
    return {
        "status": "healthy" if bot.is_ready() else "starting",
        "service": "kimi-discord-bot",
        "version": "2.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            "discord_gateway": bot.is_ready(),
            "http_session": bot.session is not None and not bot.session.closed,
            "kimi_linear": bool(os.getenv("KIMI_LINEAR_URL")),
            "cognee": bool(os.getenv("COGNEE_URL")),
            "github_integration": bool(os.getenv("GITHUB_INTEGRATION_URL"))
        },
        "metrics": {
            "latency_ms": round(bot.latency * 1000, 2) if bot.latency else None,
            "guilds": len(bot.guilds),
            "users": sum(guild.member_count for guild in bot.guilds),
            "uptime_seconds": uptime
        }
    }

# Intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
intents.reactions = True

# NEU: Approval View für autonome Edits
class AutonomousApprovalView(View):
    def __init__(self, author_id: int, repo_url: str, prompt: str):
        super().__init__(timeout=900)
        self.author_id = author_id
        self.repo_url = repo_url
        self.prompt = prompt
        self.value = None  # ✅ NEU - Track result
    
    # ✅ NEU - Timeout Handler
    async def on_timeout(self):
        """Called when view times out"""
        for item in self.children:
            item.disabled = True
        
        if hasattr(self, 'message'):
            await self.message.edit(
                content="⏱️ Approval-Anfrage abgelaufen (15 Minuten)",
                view=self
            )
    
    @discord.ui.button(label="✅ Approve", style=discord.ButtonStyle.green)
    async def approve_callback(self, button: Button, interaction: discord.Interaction):
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("❌ Nicht autorisiert", ephemeral=True)
            return
        
        await interaction.response.send_message("🚀 Starte autonome Bearbeitung...", ephemeral=True)
        
        try:
            result = await process_autonomous_edit(self.repo_url, self.prompt, str(interaction.user.id))
            await interaction.followup.send(f"✅ {result}", ephemeral=True)
        except Exception as e:
            logger.error(f"Autonome Bearbeitung fehlgeschlagen: {e}")
            await interaction.followup.send(f"❌ Fehler: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="❌ Deny", style=discord.ButtonStyle.red)
    async def deny_callback(self, button: Button, interaction: discord.Interaction):
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("❌ Nicht autorisiert", ephemeral=True)
            return
        
        self.value = False  # ✅ NEU - Set result
        await interaction.response.send_message("❌ Anfrage abgebrochen", ephemeral=True)

class KimiBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='!',
            intents=intents,
            description="Kimi Linear 48B AI Coding Assistant",
            help_command=commands.DefaultHelpCommand()
        )
        
        self.kimi_url = os.getenv("KIMI_LINEAR_URL", "http://kimi-linear:8003")
        self.cognee_url = os.getenv("COGNEE_URL", "http://cognee:8001")
        self.github_url = os.getenv("GITHUB_INTEGRATION_URL", "http://github-integration:8004")
        self.session: Optional[aiohttp.ClientSession] = None
        self.start_time = datetime.now(timezone.utc)
        self.health_executor = None  # ✅ NEU - Executor speichern
        
    async def setup_hook(self):
        """Bot-Initialisierung"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            headers={"User-Agent": "Kimi-Bot/2.1"}
        )
        
        # Remote Control Cog laden
        await setup_remote_control(self)
        
        # HTTP Health Server starten
        def run_health_server():
            uvicorn.run(
                health_app, 
                host="0.0.0.0", 
                port=int(os.getenv("HEALTH_PORT", "8005")),
                log_level="info",
                access_log=False
            )
        
        loop = asyncio.get_event_loop()
        self.health_executor = ThreadPoolExecutor(max_workers=1)  # ✅ Speichern
        loop.run_in_executor(self.health_executor, run_health_server)
        
        logger.info("🤖 Discord Bot mit Remote Control initialisiert")
        logger.info("🌐 Health Check Server gestartet auf Port 8005")
        
    async def close(self):
        """Aufräumen beim Herunterfahren"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                logger.info("HTTP Session geschlossen")
        except Exception as e:
            logger.error(f"Error closing session: {e}")
        
        # ✅ NEU - Executor cleanup
        if self.health_executor:
            self.health_executor.shutdown(wait=True)
            logger.info("Health server executor shutdown")
        
        await super().close()

# ✅ NEU - Nach KimiBot Class Definition
class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> bool:
        now = time()
        user_requests = self.requests[user_id]
        
        # Entferne alte Requests
        user_requests[:] = [req for req in user_requests if now - req < self.window]
        
        if len(user_requests) >= self.max_requests:
            return False
        
        user_requests.append(now)
        return True
    
    def reset(self, user_id: str):
        self.requests.pop(user_id, None)

# ✅ Instanz erstellen
rate_limiter = RateLimiter(max_requests=10, window=60)

# ✅ NEU - Spezifische Error Codes
class BotError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

# Bot-Instanz
bot = KimiBot()

@bot.event
async def on_ready():
    """Bot ist bereit"""
    logger.info(f"✅ {bot.user} ist online!")
    logger.info(f"📊 Verbunden mit {len(bot.guilds)} Server(n)")
    
    # ✅ RICHTIG - Jetzt erst Commands synchronisieren
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash command(s)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")
    
    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name="AI Code Generation")
    )

@bot.event
async def on_command_error(ctx: commands.Context, error):
    """Globale Fehlerbehandlung"""
    # ✅ RICHTIG - Spezifische Error Codes
    if isinstance(error, BotError):
        embed = discord.Embed(
            title=f"❌ Fehler ({error.code})",
            description=error.message,
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
    elif isinstance(error, commands.CommandNotFound):
        return
    else:
        logger.error(f"Unhandled error: {error}", exc_info=True)
        await ctx.send("❌ Ein unerwarteter Fehler ist aufgetreten.")

# ===== CODE GENERATION =====
@bot.command(name='code')
async def generate_code(ctx: commands.Context, *, prompt: str):
    """Generiere Code mit Kimi Linear 48B"""
    # ✅ Rate Limit Check
    if not rate_limiter.is_allowed(str(ctx.author.id)):
        await ctx.send("⏱️ Rate limit erreicht. Bitte warte 60 Sekunden.")
        return
    
    # ✅ Input Validation
    if not prompt.strip():
        await ctx.send("⚠️ Bitte gib eine Beschreibung an!")
        return
    
    if len(prompt) > 2000:
        await ctx.send("❌ Prompt zu lang (max 2000 Zeichen)")
        return
    
    async with ctx.typing():
        try:
            memories = await search_memories(prompt, str(ctx.author.id))
            enhanced_prompt = build_enhanced_prompt(prompt, memories)
            response = await generate_with_kimi(enhanced_prompt)
            
            if not response:
                await ctx.send("❌ Code-Generierung fehlgeschlagen")
                return
            
            await store_memory(user_id=str(ctx.author.id), content=response["text"], prompt=prompt, 
                             tokens=response["tokens_used"], guild_id=str(ctx.guild.id) if ctx.guild else None)
            await send_code_response(ctx, response, prompt)
        except Exception as e:
            logger.error(f"Fehler bei Code-Generierung: {e}")
            await ctx.send(f"❌ Es ist ein Fehler aufgetreten: {str(e)}")

async def search_memories(query: str, user_id: str) -> list:
    """Suche nach relevanten Memories"""
    try:
        async with bot.session.get(f"{bot.cognee_url}/memory/search", params={"query": query, "user_id": user_id, "limit": 3}) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("memories", [])
    except Exception as e:
        logger.warning(f"Memory-Suche fehlgeschlagen: {e}")
    return []

def build_enhanced_prompt(prompt: str, memories: list) -> str:
    """Baut Prompt mit Kontext aus Memories"""
    context = ""
    if memories:
        context = "Previous relevant code:\n"
        for mem in memories:
            content_preview = mem['content'][:150] + "..." if len(mem['content']) > 150 else mem['content']
            context += f"- {content_preview}\n"
        context += "\n"
    return f"{context}Generate high-quality Python code for: {prompt}\n\nRequirements:\n- Include proper error handling and type hints\n- Add comprehensive docstrings\n- Follow PEP 8 style guidelines\n- Make it production-ready\n- Include example usage if appropriate\n\nCode:\n```python\n"

async def generate_with_kimi(prompt: str) -> Optional[dict]:
    """Rufe Kimi Linear API auf"""
    try:
        payload = {"prompt": prompt, "max_tokens": 2048, "temperature": 0.2, "top_p": 0.95}
        async with bot.session.post(f"{bot.kimi_url}/generate", json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                return await resp.json()
            elif resp.status == 503:
                logger.error("Model not loaded")
                return None
            else:
                logger.error(f"Kimi API error: {resp.status}")
                return None
    except aiohttp.ClientError as e:
        logger.error(f"Network error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

async def store_memory(user_id: str, content: str, prompt: str, tokens: int, guild_id: Optional[str]):
    """Speichere generierten Code in Cognee"""
    payload = {
        "user_id": user_id,
        "content": content,
        "metadata": {"type": "code_generation", "prompt": prompt, "language": "python", "tokens_used": tokens, "guild_id": guild_id, "model": "Kimi Linear 48B (A3B)"}
    }
    try:
        async with bot.session.post(f"{bot.cognee_url}/memory/store", json=payload) as resp:
            if resp.status != 200:
                logger.warning(f"Memory-Speicherung fehlgeschlagen: {resp.status}")
    except Exception as e:
        logger.warning(f"Memory-Speicherung fehlgeschlagen: {e}")

async def send_code_response(ctx: commands.Context, response: dict, original_prompt: str):
    """Sende generierten Code als Discord-Nachricht"""
    code = response["text"]
    tokens_used = response["tokens_used"]
    if not code.strip().startswith("```"):
        code = f"```python\n{code}\n```"
    
    embed = discord.Embed(title="🤖 AI-Generated Code", description=f"**Anfrage:** {original_prompt[:100]}...\n\n{code}", color=discord.Color.green(), timestamp=datetime.now(timezone.utc))
    embed.set_footer(text=f"Tokens: {tokens_used} | Modell: Kimi Linear 48B (A3B)")
    
    if len(embed.description) > 4000:
        chunks = [code[i:i+1900] for i in range(0, len(code), 1900)]
        for i, chunk in enumerate(chunks):
            chunk_embed = discord.Embed(title=f"🤖 Code (Teil {i+1}/{len(chunks)})", description=f"```python\n{chunk}\n```", color=discord.Color.green(), timestamp=datetime.now(timezone.utc))
            if i == 0:
                chunk_embed.description = f"**Anfrage:** {original_prompt[:100]}...\n\n{chunk_embed.description}"
            chunk_embed.set_footer(text=f"Tokens: {tokens_used} | Angefordert von: {ctx.author}")
            await ctx.send(embed=chunk_embed)
    else:
        await ctx.send(embed=embed)

# ===== MEMORY & SEARCH =====
@bot.command(name='search')
async def search_memory(ctx: commands.Context, *, query: str):
    """Suche in deinen gespeicherten Codes"""
    async with ctx.typing():
        try:
            results = await search_memories(query, str(ctx.author.id))
            if not results:
                embed = discord.Embed(title="🔍 Keine Ergebnisse", description=f"Nichts gefunden für: `{query}`", color=discord.Color.orange())
                await ctx.send(embed=embed)
                return
            
            embed = discord.Embed(title=f"🔍 Suchergebnisse für: {query}", color=discord.Color.blue(), timestamp=datetime.now(timezone.utc))
            for i, mem in enumerate(results[:5]):
                content = mem['content'][:200] + "..." if len(mem['content']) > 200 else mem['content']
                similarity = round(mem['similarity'] * 100, 1)
                embed.add_field(name=f"Ergebnis {i+1} (Ähnlichkeit: {similarity}%)", value=f"```{content}```", inline=False)
            
            embed.set_footer(text=f"{len(results)} Ergebnisse | Gesucht von: {ctx.author}")
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Fehler bei Suche: {e}")
            await ctx.send(f"❌ Suche fehlgeschlagen: {str(e)}")

@bot.command(name='mymemories')
async def my_memories(ctx: commands.Context, limit: int = 10):
    """Zeige deine gespeicherten Memories"""
    async with ctx.typing():
        try:
            async with bot.session.get(f"{bot.cognee_url}/memory/user/{ctx.author.id}", params={"limit": limit}) as resp:
                if resp.status != 200:
                    await ctx.send("❌ Fehler beim Abrufen der Memories")
                    return
                data = await resp.json()
                memories = data.get("memories", [])
            
            if not memories:
                embed = discord.Embed(title="📭 Keine Memories", description="Du hast noch keine gespeicherten Codes.", color=discord.Color.orange())
                await ctx.send(embed=embed)
                return
            
            embed = discord.Embed(title=f"📚 Deine Memories ({len(memories)})", color=discord.Color.purple(), timestamp=datetime.now(timezone.utc))
            for mem in memories:
                content_preview = mem['content'][:100] + "..." if len(mem['content']) > 100 else mem['content']
                created_at = datetime.fromisoformat(str(mem['created_at'])).strftime("%d.%m.%Y %H:%M")
                embed.add_field(name=f"Memory #{mem['id']} ({created_at})", value=content_preview, inline=False)
            
            embed.set_footer(text=f"Angefordert von: {ctx.author}")
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen: {e}")
            await ctx.send(f"❌ Fehler: {str(e)}")

@bot.command(name='feedback')
async def feedback_command(ctx: commands.Context, memory_id: int, feedback: str, *, reason: str = ""):
    """Speichere Feedback: !feedback <memory_id> <up|down|edit> <Grund>"""
    feedback_type = {"up": "thumbs_up", "down": "thumbs_down", "edit": "edited"}.get(feedback, "unknown")
    payload = {
        "user_id": str(ctx.author.id),
        "content": f"Feedback für Memory {memory_id}: {feedback_type}",
        "metadata": {"type": "user_feedback", "memory_id": memory_id, "feedback_type": feedback_type, "reason": reason, "timestamp": datetime.now().isoformat()}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{bot.cognee_url}/memory/store", json=payload) as resp:
            if resp.status == 201:
                await ctx.send("✅ Feedback gespeichert - hilft mir besser zu werden!")
            else:
                await ctx.send("❌ Feedback speichern fehlgeschlagen")

# ===== GITHUB INTEGRATION =====
@bot.command(name='github')
async def github_integration(ctx: commands.Context, action: str, *, args: str = ""):
    """GitHub Integration Commands: !github clone <url> | !github analyze <repo> | !github status"""
    action = action.lower()
    if action == "clone":
        await clone_repo_command(ctx, args)
    elif action == "analyze":
        await analyze_repo_command(ctx, args)
    elif action == "status":
        await github_status_command(ctx)
    else:
        embed = discord.Embed(title="🔧 GitHub Integration", description="**Verfügbare Commands:**\n`!github clone <url>` - Klone und analysiere Repository\n`!github analyze <repo>` - Analysiere geklontes Repository\n`!github status` - Zeigt GitHub Rate Limit", color=discord.Color.blue())
        await ctx.send(embed=embed)

async def clone_repo_command(ctx: commands.Context, url: str):
    """Klone Repository im Hintergrund"""
    if not url.startswith("https://github.com/"):
        await ctx.send("❌ Ungültige GitHub URL. Format: `https://github.com/user/repo`")
        return
    
    async with ctx.typing():
        payload = {"repo_url": url, "branch": "main", "user_id": str(ctx.author.id), "auto_analyze": True}
        try:
            async with bot.session.post(f"{bot.github_url}/repo/clone", json=payload) as resp:
                if resp.status == 202:
                    data = await resp.json()
                    embed = discord.Embed(title="🚀 Repository wird geklont", description=f"**Repo:** {url}\n**ID:** {data['repo_id']}\nStatus: Im Hintergrund verarbeitet...", color=discord.Color.green())
                    await ctx.send(embed=embed)
                else:
                    await ctx.send("❌ Clone fehlgeschlagen")
        except Exception as e:
            logger.error(f"GitHub Clone Fehler: {e}")
            await ctx.send(f"❌ Fehler: {str(e)}")

async def analyze_repo_command(ctx: commands.Context, repo: str):
    """Analysiere geklontes Repository"""
    async with ctx.typing():
        try:
            async with bot.session.post(f"{bot.github_url}/repo/analyze", json={"repo_id": repo, "user_id": str(ctx.author.id)}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embed = discord.Embed(title=f"📊 Analyse: {repo}", description=data.get("summary", "Analyse abgeschlossen"), color=discord.Color.green())
                    await ctx.send(embed=embed)
                else:
                    await ctx.send(f"❌ Analyse fehlgeschlagen: {resp.status}")
        except Exception as e:
            logger.error(f"GitHub Analyse Fehler: {e}")
            await ctx.send(f"❌ Fehler: {str(e)}")

async def github_status_command(ctx: commands.Context):
    """Zeige GitHub Rate Limit"""
    try:
        async with bot.session.get(f"{bot.github_url}/health") as resp:
            data = await resp.json()
            
        embed = discord.Embed(title="📊 GitHub Integration Status", color=discord.Color.blue())
        embed.add_field(name="Authentifiziert", value="✅ Ja" if data["github_authenticated"] else "❌ Nein (nur Public Repos)", inline=False)
        
        if "rate_limit" in data:
            rl = data["rate_limit"]
            embed.add_field(name="API Rate Limit", value=f"{rl['remaining']}/{rl['limit']} verbleibend", inline=False)
        
        await ctx.send(embed=embed)
    except Exception as e:
        logger.error(f"GitHub Status Fehler: {e}")
        await ctx.send("❌ Status abrufen fehlgeschlagen")

# ===== AUTONOMOUS EDITING =====
@bot.command(name='autonomous')
async def autonomous_edit(ctx: commands.Context, repo_url: str, *, prompt: str):
    """Autonome Repository Bearbeitung mit Approval-Flow"""
    if not prompt.strip():
        await ctx.send("❌ Bitte gib eine Anfrage an (z.B. 'Add error handling to all API endpoints')")
        return
    
    view = AutonomousApprovalView(ctx.author.id, repo_url, prompt)
    embed = discord.Embed(
        title="🤖 Autonome Änderung - Bestätigung erforderlich", 
        color=discord.Color.yellow(),
        timestamp=datetime.now(timezone.utc)
    )
    embed.add_field(name="Repository", value=repo_url, inline=False)
    embed.add_field(name="Anfrage", value=prompt, inline=False)
    embed.set_footer(text="Nur der Anfragende kann approve/deny")
    
    # ✅ NEU - Speichere Message Referenz für Timeout
    message = await ctx.send(embed=embed, view=view)
    view.message = message

# NEU: Slash Command als Alternative
@bot.slash_command(name="autonomous", description="Starte autonome Repository-Änderung (mit Approval)")
async def autonomous_slash(
    ctx: discord.ApplicationContext,
    repo_url: str,
    prompt: str
):
    """Discord Slash Command mit Approval-Flow"""
    if not ctx.guild:
        await ctx.respond("❌ Nur in Servern verfügbar", ephemeral=True)
        return
    
    view = AutonomousApprovalView(ctx.author.id, repo_url, prompt)
    embed = discord.Embed(title="🤖 Autonome Änderung", color=discord.Color.blue())
    embed.add_field(name="Repository", value=repo_url, inline=False)
    embed.add_field(name="Anfrage", value=prompt, inline=False)
    
    await ctx.respond(embed=embed, view=view, ephemeral=True)

async def process_autonomous_edit(repo_url: str, prompt: str, user_id: str) -> str:
    """Neue Hilfsfunktion: Führt die autonome Bearbeitung aus"""
    payload = {"prompt": prompt, "repo_url": repo_url, "user_id": user_id}
    try:
        async with bot.session.post(
            f"{bot.github_url}/autonomous/edit", 
            json=payload, 
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            if resp.status == 202:
                data = await resp.json()
                status = data.get('status', 'Unbekannter Status')
                return f"Autonome Bearbeitung gestartet: {status}"
            else:
                error = await resp.json()
                return f"Fehler: {error.get('detail', 'Unbekannter Fehler')}"
    except asyncio.TimeoutError:
        return "⏱️ Timeout - Die Bearbeitung benötigt zu lange"
    except Exception as e:
        logger.error(f"Autonome Bearbeitung Fehler: {e}")
        return f"Fehler: {str(e)}"

@bot.command(name='autonomous-status')
async def autonomous_status(ctx: commands.Context, job_id: str):
    """Status einer autonomen Bearbeitung abfragen"""
    await ctx.send("ℹ️ Job Tracking kommt bald...")

# ===== REACTION FEEDBACK =====
@bot.event
async def on_reaction_add(reaction: discord.Reaction, user: discord.User):
    if user.bot:
        return
    if reaction.message.author == bot.user:
        if str(reaction.emoji) == "👍":
            await store_quick_feedback(reaction.message.id, "thumbs_up", user.id)
        elif str(reaction.emoji) == "👎":
            await store_quick_feedback(reaction.message.id, "thumbs_down", user.id)

async def store_quick_feedback(message_id: int, feedback_type: str, user_id: int):
    """Speichere schnelles Reaction-Feedback"""
    logger.info(f"Feedback {feedback_type} für Message {message_id} von User {user_id}")

# ===== BOT START =====
if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token or token == "your_discord_token_here":
        logger.error("❌ Kein DISCORD_BOT_TOKEN gefunden!")
        exit(1)
    
    try:
        bot.run(token, log_handler=None)
    except Exception as e:
        logger.error(f"Bot konnte nicht starten: {e}")
        exit(1)