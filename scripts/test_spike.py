import asyncio
import aiohttp
import time

URL = "http://localhost:8000/predict"
# Un payload valide pour ne pas déclencher l'alerte des erreurs 500 en même temps
PAYLOAD = {"data": {"Pays beneficiaire": "France", "Secteur": "Santé"}}

async def send_request(session):
    try:
        async with session.post(URL, json=PAYLOAD) as response:
            return response.status
    except Exception:
        return 0

async def main():
    print("🚀 Début de l'attaque de charge (Spike)...")
    
    async with aiohttp.ClientSession() as session:
        # On prépare 150 requêtes à envoyer d'un coup
        tasks = [send_request(session) for _ in range(150)]
        
        start_time = time.time()
        # On lance tout en parallèle
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"🏁 {len(results)} requêtes envoyées en {end_time - start_time:.2f} secondes !")
        print("L'alerte 'Alerte-Trafic-Spike' va passer au rouge.")

if __name__ == "__main__":
    asyncio.run(main())