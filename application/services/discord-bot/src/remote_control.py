# services/discord-bot/remote_control.py
"""
Discord Remote Control für Kimi Linear VM
Starte/Stoppe alles über Discord Commands
"""

import discord
from discord.ext import commands, tasks
import aiohttp
import asyncio
import subprocess
import psutil
import json
import os
import time
import logging
import gc
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
import shlex

logger = logging.getLogger(__name__)

# Konstanten
COMPOSE_PATH = "/home/ubuntu/kimi-linear-complete/application"
ALLOWED_DOCKER_ACTIONS = {"start", "stop", "restart"}

class RemoteControl(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.control_channel = None
        self.allowed_users: Set[int] = set()
        self._session: Optional[aiohttp.ClientSession] = None
        self.status_monitor.start()

    @property
    def session(self) -> aiohttp.ClientSession:
        """Lazy-loaded aiohttp Session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    @commands.Cog.listener()
    async def on_ready(self):
        """Bot ist bereit - initialisiere Remote Control"""
        try:
            allowed_ids = os.getenv("DISCORD_CONTROL_USERS", "").split(",")
            self.allowed_users = {int(uid.strip()) for uid in allowed_ids if uid.strip()}
            
            channel_id = os.getenv("DISCORD_CONTROL_CHANNEL")
            if channel_id:
                self.control_channel = self.bot.get_channel(int(channel_id))
                
            if self.control_channel:
                embed = discord.Embed(
                    title="🎮 Remote Control Aktiv",
                    description="Kimi Linear VM kann über Discord gesteuert werden",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                await self.control_channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Fehler bei on_ready: {e}")

    def is_authorized(self, user_id: int) -> bool:
        """Prüfe ob User Remote-Control darf"""
        return user_id in self.allowed_users or str(user_id) == os.getenv("DISCORD_OWNER_ID")

    async def cog_before_invoke(self, ctx: commands.Context):
        """Prüfe Authentifizierung vor jedem Command"""
        if not self.is_authorized(ctx.author.id):
            raise commands.CheckFailure("❌ Keine Berechtigung für Remote-Control")
        return True

    # ===== BASIS COMMANDS =====
    @commands.command(name='vmstart')
    async def vm_start(self, ctx: commands.Context):
        """Starte komplette Kimi Linear Engine"""
        async with ctx.typing():
            try:
                result = await self.start_all_services()
                embed = discord.Embed(
                    title="🚀 VM Start Initiiert",
                    description=result["message"],
                    color=discord.Color.green() if result["success"] else discord.Color.red(),
                    timestamp=datetime.now()
                )
                embed.add_field(name="Services", value=result["services_started"], inline=False)
                embed.add_field(name="Dauer", value=f"{result['duration']}s")
                if result["logs"]:
                    embed.add_field(name="Logs", value=f"```{result['logs'][:500]}```", inline=False)
                await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Fehler beim Starten: {e}")
                await ctx.send(f"❌ Start fehlgeschlagen: {str(e)}")

    @commands.command(name='vmstop')
    async def vm_stop(self, ctx: commands.Context):
        """Stoppe alle Services"""
        async with ctx.typing():
            try:
                result = await self.stop_all_services()
                embed = discord.Embed(
                    title="⏹️ VM Stop Initiiert",
                    description=result["message"],
                    color=discord.Color.orange(),
                    timestamp=datetime.now()
                )
                embed.add_field(name="Services Gestoppt", value=result["services_stopped"])
                await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Fehler beim Stoppen: {e}")
                await ctx.send(f"❌ Stop fehlgeschlagen: {str(e)}")

    @commands.command(name='vmstatus')
    async def vm_status(self, ctx: commands.Context):
        """Zeige kompletten VM-Status"""
        async with ctx.typing():
            try:
                status = await self.get_full_status()
                embed = discord.Embed(title="📊 VM Status Report", color=discord.Color.blue(), timestamp=datetime.now())
                
                embed.add_field(name="🖥️ System", value=f"CPU: {status['cpu']:.1f}%\nRAM: {status['memory']:.1f}%\nDisk: {status['disk']:.1f}%", inline=True)
                
                if status.get('gpu'):
                    gpu = status['gpu']
                    embed.add_field(
                        name="🎮 GPU", 
                        value=f"Name: {gpu['name']}\nVRAM: {gpu['memory_used']}/{gpu['memory_total']} GB\nUtil: {gpu['utilization']}%", 
                        inline=True
                    )
                
                services_status = [f"{'🟢' if info['running'] else '🔴'} {service}: {info['status']}" for service, info in status['services'].items()]
                embed.add_field(name="🔧 Services", value="\n".join(services_status) or "Keine Services", inline=False)
                
                if status.get('endpoints'):
                    endpoints = []
                    for name, url in status['endpoints'].items():
                        healthy = await self.check_url_health(url)
                        endpoints.append(f"{'🟢' if healthy else '🔴'} [{name}]({url})")
                    embed.add_field(name="🌐 Endpunkte", value="\n".join(endpoints), inline=False)
                
                embed.set_footer(text=f"Angefordert von: {ctx.author}")
                await ctx.send(embed=embed)
            except Exception as e:
                logger.error(f"Fehler beim Abrufen des Status: {e}")
                await ctx.send(f"❌ Status konnte nicht abgerufen werden: {str(e)}")

    # ===== ERWEITERTE COMMANDS =====
    @commands.command(name='vmservices')
    async def vm_services(self, ctx: commands.Context, action: str = "list", service: str = None):
        """Verwalte einzelne Services (start/stop/restart/list/logs)"""
        actions = ["start", "stop", "restart", "list", "logs"]
        
        if action not in actions:
            await ctx.send(f"❌ Ungültige Aktion. Verfügbar: {', '.join(actions)}")
            return
            
        # Prüfe Service-Name für Aktionen, die einen Service benötigen
        if action in ["start", "stop", "restart", "logs"] and not service:
            await ctx.send(f"❌ Service-Name erforderlich für Aktion: {action}")
            return
        
        try:
            async with ctx.typing():
                if action == "list":
                    services = await self.get_service_list()
                    embed = discord.Embed(
                        title="📋 Verfügbare Services", 
                        description="\n".join([f"• `{s}`" for s in services]) or "Keine Services gefunden",
                        color=discord.Color.blue()
                    )
                    await ctx.send(embed=embed)
                    
                elif action == "logs":
                    # ✅ Service-Validierung hinzugefügt
                    allowed_services = await self.get_service_list()
                    if service not in allowed_services:
                        raise ValueError(f"Ungültiger Service: `{service}`. Verfügbar: {', '.join(allowed_services)}")
                    
                    logs = await self.get_service_logs(service, lines=20)
                    embed = discord.Embed(
                        title=f"📝 Logs: {service}", 
                        description=f"```{logs[:1900]}```", 
                        color=discord.Color.greyple(),
                        timestamp=datetime.now()
                    )
                    await ctx.send(embed=embed)
                    
                else:
                    # start, stop, restart
                    result = await self.manage_service(action, service)
                    embed = discord.Embed(
                        title=f"🔧 Service {action.title()}: {service}", 
                        description=result["message"], 
                        color=discord.Color.green() if result["success"] else discord.Color.red(),
                        timestamp=datetime.now()
                    )
                    await ctx.send(embed=embed)
                    
        except ValueError as e:
            await ctx.send(f"❌ {str(e)}")
        except Exception as e:
            logger.error(f"Fehler in vm_services ({action}, {service}): {e}")
            await ctx.send(f"❌ Fehlgeschlagen: {str(e)}")

    @commands.command(name='vmconfig')
    async def vm_config(self, ctx: commands.Context, key: str = None, value: str = None):
        """Zeige oder ändere VM-Konfiguration"""
        async with ctx.typing():
            try:
                if key and value:
                    # Sichere Eingabevalidierung
                    if not self._is_valid_env_key(key):
                        await ctx.send("❌ Ungültiger Config-Schlüssel")
                        return
                    result = await self.set_config(key, value)
                    if result:
                        await ctx.send(f"✅ Config gesetzt: `{key}` = `{value}`")
                    else:
                        await ctx.send("❌ Fehler beim Setzen des Config-Werts")
                        
                elif key:
                    val = await self.get_config(key)
                    await ctx.send(f"🔍 `{key}` = `{val}`")
                    
                else:
                    configs = await self.get_all_configs()
                    if configs:
                        config_lines = []
                        for k, v in configs.items():
                            # Sensible Daten maskieren
                            if any(secret in k.lower() for secret in ['token', 'password', 'key', 'secret']):
                                v = "***"
                            config_lines.append(f"`{k}` = `{v}`")
                        
                        embed = discord.Embed(
                            title="⚙️ VM Konfiguration", 
                            description="\n".join(config_lines), 
                            color=discord.Color.blue()
                        )
                        await ctx.send(embed=embed)
                    else:
                        await ctx.send("Keine Konfiguration gefunden")
                        
            except Exception as e:
                logger.error(f"Fehler in vm_config: {e}")
                await ctx.send(f"❌ Fehler: {str(e)}")

    @commands.command(name='vmmonitor')
    async def vm_monitor(self, ctx: commands.Context, duration: int = 60):
        """Starte Live-Monitoring für X Sekunden"""
        if duration > 300:  # Max 5 Minuten
            await ctx.send("❌ Maximale Dauer: 300 Sekunden")
            return
            
        embed = discord.Embed(
            title="📈 Live-Monitor Aktiv", 
            description=f"Monitoring für {duration}s...", 
            color=discord.Color.green()
        )
        message = await ctx.send(embed=embed)
        
        try:
            for i in range(0, duration, 10):
                status = await self.get_quick_status()
                embed.description = (
                    f"⏱️ Laufzeit: {i}s/{duration}s\n"
                    f"🖥️ CPU: {status['cpu']:.1f}%\n"
                    f"🧠 RAM: {status['memory']:.1f}%\n"
                    f"💾 Disk: {status['disk']:.1f}%"
                )
                
                # GPU nur anzeigen, wenn verfügbar
                if status.get('gpu'):
                    gpu = status['gpu']
                    embed.add_field(
                        name="🎮 GPU", 
                        value=f"VRAM: {gpu['memory_used']}/{gpu['memory_total']} GB", 
                        inline=False
                    )
                
                embed.timestamp = datetime.now()
                await message.edit(embed=embed)
                await asyncio.sleep(10)
            
            embed.title = "✅ Monitoring Abgeschlossen"
            embed.color = discord.Color.blue()
            await message.edit(embed=embed)
            
        except asyncio.CancelledError:
            embed.title = "⏹️ Monitoring Abgebrochen"
            embed.color = discord.Color.orange()
            await message.edit(embed=embed)
        except Exception as e:
            logger.error(f"Fehler in vm_monitor: {e}")
            embed.title = "❌ Monitoring Fehlgeschlagen"
            embed.color = discord.Color.red()
            await message.edit(embed=embed)

    @commands.command(name='vmauto')
    async def vm_auto(self, ctx: commands.Context, action: str = "status"):
        """Verwalte automatische Neustarts/Monitoring"""
        action = action.lower()
        
        if action == "on":
            if not self.status_monitor.is_running():
                self.status_monitor.start()
                await ctx.send("🤖 Automatisches Monitoring aktiviert")
            else:
                await ctx.send("⚠️ Automatisches Monitoring war bereits aktiv")
                
        elif action == "off":
            if self.status_monitor.is_running():
                self.status_monitor.stop()
                await ctx.send("⏹️ Automatisches Monitoring deaktiviert")
            else:
                await ctx.send("⚠️ Automatisches Monitoring war bereits inaktiv")
                
        else:
            status = "🟢 Aktiv" if self.status_monitor.is_running() else "🔴 Inaktiv"
            await ctx.send(f"📊 Auto-Monitoring: {status}")

    @tasks.loop(minutes=5)
    async def status_monitor(self):
        """Automatischer Status-Check alle 5 Minuten"""
        if not self.control_channel:
            return
        
        try:
            status = await self.get_quick_status()
            
            # Warnung bei hoher Auslastung
            if status['cpu'] > 90 or status['memory'] > 85:
                embed = discord.Embed(
                    title="⚠️ Hohe Auslastung erkannt!",
                    description=f"CPU: {status['cpu']:.1f}%\nRAM: {status['memory']:.1f}%",
                    color=discord.Color.orange(),
                    timestamp=datetime.now()
                )
                await self.control_channel.send(embed=embed)
                
            # Warnung bei fast voller Disk
            if status['disk'] > 90:
                embed = discord.Embed(
                    title="⚠️ Hohe Disk-Auslastung!",
                    description=f"Disk: {status['disk']:.1f}%",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                await self.control_channel.send(embed=embed)
                
        except Exception as e:
            logger.error(f"Status monitor error: {e}")
        finally:
            gc.collect()

    @status_monitor.before_loop
    async def before_status_monitor(self):
        await self.bot.wait_until_ready()

    # ===== TECHNISCHE FUNKTIONEN =====
    async def _safe_docker_command(
        self,
        cmd: List[str],
        timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """Sichere docker-compose Ausführung mit Timeout"""
        if not all(isinstance(arg, str) for arg in cmd):
            raise ValueError("Alle Kommando-Argumente müssen Strings sein")
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    cwd=COMPOSE_PATH,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False
                )
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Command failed: {e}")
            raise

    async def start_all_services(self) -> Dict[str, Any]:
        """Starte alle Services in korrekter Reihenfolge"""
        start_time = time.time()
        try:
            result = await self._safe_docker_command(["docker-compose", "up", "-d"])
            return {
                "success": result.returncode == 0,
                "message": "Alle Services erfolgreich gestartet" if result.returncode == 0 else "Fehler beim Starten",
                "services_started": "postgres, redis, kimi-linear, cognee, discord-bot, github-integration, prometheus, grafana",
                "duration": int(time.time() - start_time),
                "logs": result.stdout if result.returncode == 0 else result.stderr
            }
        except Exception as e:
            logger.error(f"Fehler in start_all_services: {e}")
            return {"success": False, "message": f"Fehler: {str(e)}", "services_started": "N/A", "duration": 0, "logs": ""}

    async def stop_all_services(self) -> Dict[str, Any]:
        """Stoppe alle Services"""
        try:
            result = await self._safe_docker_command(["docker-compose", "down"])
            return {
                "success": result.returncode == 0,
                "message": "Services gestoppt" if result.returncode == 0 else "Fehler beim Stoppen",
                "services_stopped": "Alle Container",
                "logs": result.stdout if result.returncode == 0 else result.stderr
            }
        except Exception as e:
            logger.error(f"Fehler in stop_all_services: {e}")
            return {"success": False, "message": f"Fehler: {str(e)}", "services_stopped": "N/A", "logs": ""}

    async def get_full_status(self) -> Dict[str, Any]:
        """Hole kompletten System-Status"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            gpu_info = None
            try:
                result = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                if result.returncode == 0:
                    parts = stdout.decode().strip().split(', ')
                    if len(parts) >= 4:
                        gpu_info = {"name": parts[0], "memory_used": int(parts[1]) // 1024, "memory_total": int(parts[2]) // 1024, "utilization": int(parts[3])}
            except Exception as e:
                logger.debug(f"GPU Status nicht verfügbar: {e}")
            
            services = {}
            try:
                result = await self._safe_docker_command(["docker-compose", "ps", "--format", "json"])
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            service_data = json.loads(line)
                            services[service_data['Service']] = {"running": service_data['State'] == 'running', "status": service_data['Status']}
            except Exception as e:
                logger.error(f"Fehler beim Abrufen der Services: {e}")
                services = {"error": {"running": False, "status": "Nicht erreichbar"}}
            
            endpoints = {
                "Kimi API": "http://localhost:8003/health",
                "Cognee API": "http://localhost:8001/health", 
                "Grafana": "http://localhost:3000/api/health",
                "Prometheus": "http://localhost:9090/-/healthy"
            }
            
            return {
                "cpu": cpu, 
                "memory": memory.percent, 
                "disk": (disk.used / disk.total) * 100, 
                "gpu": gpu_info, 
                "services": services, 
                "endpoints": endpoints
            }
        except Exception as e:
            logger.error(f"Fehler in get_full_status: {e}")
            return {"cpu": 0, "memory": 0, "disk": 0, "gpu": None, "services": {"error": {"running": False, "status": "Status nicht verfügbar"}}, "endpoints": {}}

    async def get_quick_status(self) -> Dict[str, Any]:
        """Schneller Status für Monitoring"""
        try:
            cpu = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory().percent
            disk = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            
            gpu_info = None
            try:
                result = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                if result.returncode == 0:
                    parts = stdout.decode().strip().split(', ')
                    if len(parts) >= 2:
                        gpu_info = {"memory_used": int(parts[0]) // 1024, "memory_total": int(parts[1]) // 1024}
            except:
                pass
            
            return {"cpu": cpu, "memory": memory, "disk": disk, "gpu": gpu_info}
        except Exception as e:
            logger.error(f"Fehler in get_quick_status: {e}")
            return {"cpu": 0, "memory": 0, "disk": 0, "gpu": None}

    async def check_url_health(self, url: str) -> bool:
        """Prüfe ob URL erreichbar ist"""
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False

    async def get_service_logs(self, service: str, lines: int = 50) -> str:
        """Hole Logs eines Services"""
        try:
            result = await self._safe_docker_command(["docker-compose", "logs", "--tail", str(lines), service])
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Logs für {service}: {e}")
            return f"Fehler: {str(e)}"

    async def manage_service(self, action: str, service: str) -> Dict[str, Any]:
        """Verwalte einzelnen Service (start/stop/restart)"""
        # ✅ Vereinfachte Validierung wie im Snippet vorgeschlagen
        if action not in ALLOWED_DOCKER_ACTIONS:
            raise ValueError(f"Ungültige Aktion: {action}")
        
        allowed_services = await self.get_service_list()
        if service and service not in allowed_services:
            raise ValueError(f"Ungültiger Service: {service}")
        
        try:
            result = await self._safe_docker_command(["docker-compose", action, service])
            return {"success": result.returncode == 0, "message": result.stdout or result.stderr}
        except Exception as e:
            logger.error(f"Fehler in manage_service ({action}, {service}): {e}")
            return {"success": False, "message": f"Fehler: {str(e)}"}

    async def get_service_list(self) -> List[str]:
        """Hole Liste aller Services"""
        try:
            result = await self._safe_docker_command(["docker-compose", "config", "--services"])
            if result.returncode == 0:
                return [s.strip() for s in result.stdout.strip().split('\n') if s.strip()]
            return []
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Service-Liste: {e}")
            return []

    def _is_valid_env_key(self, key: str) -> bool:
        """Prüfe ob ein ENV-Schlüssel gültig ist (einfache Injection-Vermeidung)"""
        return bool(key) and all(c.isalnum() or c == '_' for c in key) and key.isupper()

    async def set_config(self, key: str, value: str) -> bool:
        """Setze Config-Wert in .env"""
        try:
            if not self._is_valid_env_key(key):
                logger.warning(f"Ungültiger ENV-Schlüssel versucht: {key}")
                return False
            
            safe_value = shlex.quote(value)
            check_result = await self._safe_docker_command(["grep", f"^{key}=", ".env"])
            
            if check_result.returncode == 0:
                result = await self._safe_docker_command(["sed", "-i", f"s/^{key}=.*/{key}={safe_value}/", ".env"])
            else:
                result = await self._safe_docker_command(["bash", "-c", f'echo "{key}={safe_value}" >> .env'])
            
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Fehler beim Setzen der Config {key}={value}: {e}")
            return False

    async def get_config(self, key: str) -> str:
        """Hole Config-Wert aus .env"""
        try:
            result = await self._safe_docker_command(["grep", f"^{key}=", ".env"])
            if result.returncode == 0:
                return result.stdout.strip().split('=', 1)[1]
            return "Nicht gefunden"
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Config {key}: {e}")
            return "Fehler beim Lesen"

    async def get_all_configs(self) -> Dict[str, str]:
        """Hole alle Configs"""
        try:
            result = await self._safe_docker_command(["cat", ".env"])
            configs = {}
            for line in result.stdout.strip().split('\n'):
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    configs[k] = v
            return configs
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Configs: {e}")
            return {"Fehler": "Konnte .env nicht lesen"}

    async def cog_unload(self):
        """Cleanup beim Unload"""
        self.status_monitor.stop()
        if self._session and not self._session.closed:
            await self._session.close()

# Setup Funktion
async def setup_remote_control(bot):
    """Füge Remote Control zum Bot hinzu"""
    await bot.add_cog(RemoteControl(bot))