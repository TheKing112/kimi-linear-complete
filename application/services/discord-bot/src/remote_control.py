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
from datetime import datetime
from typing import Optional, Dict, Any, List

class RemoteControl(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.control_channel = None
        self.allowed_users = set()
        self.setup_complete = False
        self.status_monitor.start()

    @commands.Cog.listener()
    async def on_ready(self):
        """Bot ist bereit - initialisiere Remote Control"""
        allowed_ids = os.getenv("DISCORD_CONTROL_USERS", "").split(",")
        self.allowed_users = {int(uid.strip()) for uid in allowed_ids if uid.strip()}
        
        if os.getenv("DISCORD_CONTROL_CHANNEL"):
            self.control_channel = self.bot.get_channel(int(os.getenv("DISCORD_CONTROL_CHANNEL")))
            
        if self.control_channel:
            embed = discord.Embed(
                title="🎮 Remote Control Aktiv",
                description="Kimi Linear VM kann über Discord gesteuert werden",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            await self.control_channel.send(embed=embed)

    def is_authorized(self, user_id: int) -> bool:
        """Prüfe ob User Remote-Control darf"""
        return user_id in self.allowed_users or str(user_id) == os.getenv("DISCORD_OWNER_ID")

    # ===== BASIS COMMANDS =====
    @commands.command(name='vmstart')
    async def vm_start(self, ctx):
        """Starte komplette Kimi Linear Engine"""
        if not self.is_authorized(ctx.author.id):
            await ctx.send("❌ Keine Berechtigung für Remote-Control")
            return
            
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
                await ctx.send(f"❌ Start fehlgeschlagen: {str(e)}")

    @commands.command(name='vmstop')
    async def vm_stop(self, ctx):
        """Stoppe alle Services"""
        if not self.is_authorized(ctx.author.id):
            await ctx.send("❌ Keine Berechtigung für Remote-Control")
            return
            
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
                await ctx.send(f"❌ Stop fehlgeschlagen: {str(e)}")

    @commands.command(name='vmstatus')
    async def vm_status(self, ctx):
        """Zeige kompletten VM-Status"""
        async with ctx.typing():
            try:
                status = await self.get_full_status()
                embed = discord.Embed(title="📊 VM Status Report", color=discord.Color.blue(), timestamp=datetime.now())
                
                embed.add_field(name="🖥️ System", value=f"CPU: {status['cpu']}%\nRAM: {status['memory']}%\nDisk: {status['disk']}", inline=True)
                
                if status['gpu']:
                    gpu = status['gpu']
                    embed.add_field(name="🎮 GPU", value=f"Name: {gpu['name']}\nVRAM: {gpu['memory_used']}/{gpu['memory_total']} GB\nUtil: {gpu['utilization']}%", inline=True)
                
                services_status = [f"{'🟢' if info['running'] else '🔴'} {service}: {info['status']}" for service, info in status['services'].items()]
                embed.add_field(name="🔧 Services", value="\n".join(services_status) or "Keine Services", inline=False)
                
                if status['endpoints']:
                    endpoints = [f"{'🟢' if await self.check_url_health(url) else '🔴'} [{name}]({url})" for name, url in status['endpoints'].items()]
                    embed.add_field(name="🌐 Endpunkte", value="\n".join(endpoints), inline=False)
                
                embed.set_footer(text=f"Angefordert von: {ctx.author}")
                await ctx.send(embed=embed)
            except Exception as e:
                await ctx.send(f"❌ Status konnte nicht abgerufen werden: {str(e)}")

    # ===== ERWEITERTE COMMANDS =====
    @commands.command(name='vmservices')
    async def vm_services(self, ctx, action: str = "list", service: str = None):
        """Verwalte einzelne Services (start/stop/restart/list/logs)"""
        if not self.is_authorized(ctx.author.id):
            await ctx.send("❌ Keine Berechtigung")
            return
            
        actions = ["start", "stop", "restart", "list", "logs"]
        if action not in actions:
            await ctx.send(f"❌ Ungültige Aktion. Verfügbar: {', '.join(actions)}")
            return
            
        async with ctx.typing():
            if action == "list":
                services = await self.get_service_list()
                embed = discord.Embed(title="📋 Verfügbare Services", description="\n".join([f"• `{s}`" for s in services]), color=discord.Color.blue())
                await ctx.send(embed=embed)
            elif action == "logs" and service:
                logs = await self.get_service_logs(service, lines=20)
                embed = discord.Embed(title=f"📝 Logs: {service}", description=f"```{logs[:1900]}```", color=discord.Color.greyple())
                await ctx.send(embed=embed)
            else:
                result = await self.manage_service(action, service)
                embed = discord.Embed(title=f"🔧 Service {action.title()}: {service}", description=result["message"], color=discord.Color.green() if result["success"] else discord.Color.red())
                await ctx.send(embed=embed)

    @commands.command(name='vmconfig')
    async def vm_config(self, ctx, key: str = None, value: str = None):
        """Zeige oder ändere VM-Konfiguration"""
        if not self.is_authorized(ctx.author.id):
            await ctx.send("❌ Keine Berechtigung")
            return
            
        async with ctx.typing():
            if key and value:
                result = await self.set_config(key, value)
                await ctx.send(f"✅ Config gesetzt: {key} = {value}")
            elif key:
                val = await self.get_config(key)
                await ctx.send(f"🔍 {key} = `{val}`")
            else:
                configs = await self.get_all_configs()
                embed = discord.Embed(title="⚙️ VM Konfiguration", description="\n".join([f"`{k}` = `{v}`" for k, v in configs.items()]), color=discord.Color.blue())
                await ctx.send(embed=embed)

    @commands.command(name='vmmonitor')
    async def vm_monitor(self, ctx, duration: int = 60):
        """Starte Live-Monitoring für X Sekunden"""
        if not self.is_authorized(ctx.author.id):
            await ctx.send("❌ Keine Berechtigung")
            return
            
        embed = discord.Embed(title="📈 Live-Monitor Aktiv", description=f"Monitoring für {duration}s...", color=discord.Color.green())
        message = await ctx.send(embed=embed)
        
        for i in range(0, duration, 10):
            status = await self.get_quick_status()
            embed.description = f"⏱️ Laufzeit: {i}s/{duration}s\n🖥️ CPU: {status['cpu']}%\n🧠 RAM: {status['memory']}%\n💾 Disk: {status['disk']}"
            if status.get('gpu'):
                embed.add_field(name="🎮 GPU", value=f"VRAM: {status['gpu']['memory_used']}/{status['gpu']['memory_total']} GB", inline=False)
            await message.edit(embed=embed)
            await asyncio.sleep(10)
        
        embed.title = "✅ Monitoring Abgeschlossen"
        embed.color = discord.Color.blue()
        await message.edit(embed=embed)

    @commands.command(name='vmauto')
    async def vm_auto(self, ctx, action: str = "status"):
        """Verwalte automatische Neustarts/Monitoring"""
        if not self.is_authorized(ctx.author.id):
            await ctx.send("❌ Keine Berechtigung")
            return
            
        if action == "on":
            self.status_monitor.start()
            await ctx.send("🤖 Automatisches Monitoring aktiviert")
        elif action == "off":
            self.status_monitor.stop()
            await ctx.send("⏹️ Automatisches Monitoring deaktiviert")
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
            if status['cpu'] > 90 or status['memory'] > 85:
                embed = discord.Embed(title="⚠️ Hohe Auslastung erkannt!", description=f"CPU: {status['cpu']}%\nRAM: {status['memory']}%", color=discord.Color.orange(), timestamp=datetime.now())
                await self.control_channel.send(embed=embed)
        except Exception as e:
            print(f"Status-Monitor Fehler: {e}")

    @status_monitor.before_loop
    async def before_status_monitor(self):
        await self.bot.wait_until_ready()

    # ===== TECHNISCHE FUNKTIONEN =====
    async def start_all_services(self) -> Dict[str, Any]:
        """Starte alle Services in korrekter Reihenfolge"""
        start_time = time.time()
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd="/home/ubuntu/kimi-linear-complete/application",
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "message": "Alle Services erfolgreich gestartet" if result.returncode == 0 else "Fehler beim Starten",
            "services_started": "postgres, redis, kimi-linear, cognee, discord-bot, github-integration, prometheus, grafana",
            "duration": int(time.time() - start_time),
            "logs": result.stdout if result.returncode == 0 else result.stderr
        }

    async def stop_all_services(self) -> Dict[str, Any]:
        """Stoppe alle Services"""
        result = subprocess.run(
            ["docker-compose", "down"],
            cwd="/home/ubuntu/kimi-linear-complete/application",
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "message": "Services gestoppt" if result.returncode == 0 else "Fehler beim Stoppen",
            "services_stopped": "Alle Container",
            "logs": result.stdout if result.returncode == 0 else result.stderr
        }

    async def get_full_status(self) -> Dict[str, Any]:
        """Hole kompletten System-Status"""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        gpu_info = None
        
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                gpu_info = {"name": parts[0], "memory_used": int(parts[1]) // 1024, "memory_total": int(parts[2]) // 1024, "utilization": int(parts[3])}
        except:
            pass
        
        services = {}
        try:
            result = subprocess.run(["docker-compose", "ps", "--format", "json"], cwd="/home/ubuntu/kimi-linear-complete/application", capture_output=True, text=True)
            for line in result.stdout.strip().split('\n'):
                if line:
                    service_data = json.loads(line)
                    services[service_data['Service']] = {"running": service_data['State'] == 'running', "status": service_data['Status']}
        except:
            services = {"error": {"running": False, "status": "Nicht erreichbar"}}
        
        endpoints = {
            "Kimi API": "http://localhost:8003/health",
            "Cognee API": "http://localhost:8001/health", 
            "Grafana": "http://localhost:3000/api/health",
            "Prometheus": "http://localhost:9090/-/healthy"
        }
        
        return {"cpu": cpu, "memory": memory.percent, "disk": (disk.used / disk.total) * 100, "gpu": gpu_info, "services": services, "endpoints": endpoints}

    async def get_quick_status(self) -> Dict[str, Any]:
        """Schneller Status für Monitoring"""
        cpu = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory().percent
        disk = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        return {"cpu": cpu, "memory": memory, "disk": disk}

    async def check_url_health(self, url: str) -> bool:
        """Prüfe ob URL erreichbar ist"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except:
            return False

    async def get_service_logs(self, service: str, lines: int = 50) -> str:
        """Hole Logs eines Services"""
        result = subprocess.run(
            ["docker-compose", "logs", "--tail", str(lines), service],
            cwd="/home/ubuntu/kimi-linear-complete/application",
            capture_output=True,
            text=True
        )
        return result.stdout if result.returncode == 0 else result.stderr

    async def manage_service(self, action: str, service: str) -> Dict[str, Any]:
        """Starte/Stoppe/Restarte einzelnen Service"""
        cmd = [action] if action == "logs" else [action, service]
        result = subprocess.run(["docker-compose"] + cmd, cwd="/home/ubuntu/kimi-linear-complete/application", capture_output=True, text=True)
        return {"success": result.returncode == 0, "message": result.stdout if result.returncode == 0 else result.stderr}

    async def get_service_list(self) -> List[str]:
        """Hole Liste aller Services"""
        try:
            result = subprocess.run(["docker-compose", "config", "--services"], cwd="/home/ubuntu/kimi-linear-complete/application", capture_output=True, text=True)
            return result.stdout.strip().split('\n') if result.returncode == 0 else []
        except:
            return []

    async def set_config(self, key: str, value: str) -> bool:
        """Setze Config-Wert in .env"""
        try:
            subprocess.run(["sed", "-i", f"s/{key}=.*/{key}={value}/", ".env"], cwd="/home/ubuntu/kimi-linear-complete/application", capture_output=True)
            return True
        except:
            return False

    async def get_config(self, key: str) -> str:
        """Hole Config-Wert aus .env"""
        try:
            result = subprocess.run(["grep", f"^{key}=", ".env"], cwd="/home/ubuntu/kimi-linear-complete/application", capture_output=True, text=True)
            return result.stdout.strip().split('=')[1] if result.returncode == 0 else "Nicht gefunden"
        except:
            return "Fehler beim Lesen"

    async def get_all_configs(self) -> Dict[str, str]:
        """Hole alle Configs"""
        try:
            result = subprocess.run(["cat", ".env"], cwd="/home/ubuntu/kimi-linear-complete/application", capture_output=True, text=True)
            configs = {}
            for line in result.stdout.strip().split('\n'):
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    configs[k] = v
            return configs
        except:
            return {"Fehler": "Konnte .env nicht lesen"}

    async def cog_unload(self):
        """Cleanup beim Unload"""
        self.status_monitor.stop()

# Setup Funktion
async def setup_remote_control(bot):
    """Füge Remote Control zum Bot hinzu"""
    await bot.add_cog(RemoteControl(bot))