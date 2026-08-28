"""
Marketplace - маркетплейс ИИ-моделей.

Позволяет авторам публиковать своих ИИ-агентов,
устанавливать цену за использование,
а другим пользователям — покупать и запускать их.
"""

import json
import os
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Listing:
    """Публикация модели на маркетплейсе."""
    listing_id: str
    model_id: str
    author: str
    title: str
    description: str
    task: str
    model_type: str
    metrics: Dict[str, float]
    price_per_call: float        # цена за один запрос в USD
    total_calls: int = 0         # сколько раз использовали
    total_earnings: float = 0.0  # сколько заработал автор
    rating: float = 0.0          # средний рейтинг
    reviews: List[Dict] = field(default_factory=list)
    created_at: str = ""
    active: bool = True


class Marketplace:
    """
    Маркетплейс Genesis AI.
    
    Авторы публикуют модели, пользователи покупают.
    Платформа берёт комиссию 25%.
    """
    
    COMMISSION = 0.25  # 25% платформе
    
    def __init__(self, storage_dir: str = "./marketplace"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.listings_path = self.storage_dir / "listings.json"
        self.transactions_path = self.storage_dir / "transactions.json"
        
        self.listings: Dict[str, Dict] = self._load(self.listings_path)
        self.transactions: List[Dict] = self._load(self.transactions_path, as_list=True)
    
    def _load(self, path: Path, as_list: bool = False):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return [] if as_list else {}
    
    def _save(self, data, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def publish(
        self,
        author: str,
        model_id: str,
        title: str,
        description: str,
        task: str,
        model_type: str,
        metrics: Dict[str, float],
        price_per_call: float = 0.01,
    ) -> str:
        """
        Публикация модели на маркетплейсе.
        
        Args:
            author: имя автора
            model_id: ID модели из реестра
            title: название листинга
            description: описание что делает модель
            task: тип задачи
            model_type: тип модели
            metrics: метрики качества
            price_per_call: цена за вызов
            
        Returns:
            listing_id
        """
        listing_id = f"list_{model_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.listings[listing_id] = {
            "listing_id": listing_id,
            "model_id": model_id,
            "author": author,
            "title": title,
            "description": description,
            "task": task,
            "model_type": model_type,
            "metrics": metrics,
            "price_per_call": price_per_call,
            "total_calls": 0,
            "total_earnings": 0.0,
            "rating": 0.0,
            "reviews": [],
            "created_at": datetime.now().isoformat(),
            "active": True,
        }
        
        self._save(self.listings, self.listings_path)
        return listing_id
    
    def search(
        self,
        task: Optional[str] = None,
        model_type: Optional[str] = None,
        min_rating: float = 0.0,
        max_price: Optional[float] = None,
        sort_by: str = "rating",
        limit: int = 20,
    ) -> List[Dict]:
        """
        Поиск моделей на маркетплейсе.
        
        Args:
            task: фильтр по задаче
            model_type: фильтр по типу модели
            min_rating: минимальный рейтинг
            max_price: максимальная цена
            sort_by: сортировка (rating, calls, earnings, price, created)
            limit: лимит результатов
            
        Returns:
            список листингов
        """
        results = []
        for listing_id, listing in self.listings.items():
            if not listing.get("active", True):
                continue
            if task and listing["task"] != task:
                continue
            if model_type and listing["model_type"] != model_type:
                continue
            if listing.get("rating", 0) < min_rating:
                continue
            if max_price is not None and listing["price_per_call"] > max_price:
                continue
            results.append(listing)
        
        # Сортировка
        sort_keys = {
            "rating": lambda x: x.get("rating", 0),
            "calls": lambda x: x.get("total_calls", 0),
            "earnings": lambda x: x.get("total_earnings", 0),
            "price": lambda x: x.get("price_per_call", 0),
            "created": lambda x: x.get("created_at", ""),
        }
        
        key = sort_keys.get(sort_by, sort_keys["rating"])
        results.sort(key=key, reverse=(sort_by != "price"))
        
        return results[:limit]
    
    def use_model(self, listing_id: str, user: str) -> Dict:
        """
        Использование модели (платный вызов).
        
        Args:
            listing_id: ID листинга
            user: кто использует
            
        Returns:
            информация о транзакции
        """
        if listing_id not in self.listings:
            raise KeyError(f"Listing {listing_id} not found")
        
        listing = self.listings[listing_id]
        
        if not listing["active"]:
            raise ValueError("Listing is not active")
        
        price = listing["price_per_call"]
        platform_fee = price * self.COMMISSION
        author_earning = price - platform_fee
        
        # Обновляем статистику
        listing["total_calls"] += 1
        listing["total_earnings"] += author_earning
        
        # Запись транзакции
        transaction = {
            "transaction_id": f"txn_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "listing_id": listing_id,
            "user": user,
            "author": listing["author"],
            "price": price,
            "platform_fee": platform_fee,
            "author_earning": author_earning,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.transactions.append(transaction)
        self._save(self.listings, self.listings_path)
        self._save(self.transactions, self.transactions_path)
        
        return transaction
    
    def add_review(self, listing_id: str, user: str, rating: float, comment: str = "") -> Dict:
        """
        Добавление отзыва о модели.
        
        Args:
            listing_id: ID листинга
            user: кто оставляет отзыв
            rating: оценка 1-5
            comment: текст отзыва
        """
        if listing_id not in self.listings:
            raise KeyError(f"Listing {listing_id} not found")
        
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        review = {
            "user": user,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        }
        
        listing = self.listings[listing_id]
        listing["reviews"].append(review)
        
        # Пересчёт среднего рейтинга
        total = sum(r["rating"] for r in listing["reviews"])
        listing["rating"] = round(total / len(listing["reviews"]), 1)
        
        self._save(self.listings, self.listings_path)
        return review
    
    def get_author_earnings(self, author: str) -> Dict:
        """
        Статистика заработка автора.
        """
        author_listings = [
            l for l in self.listings.values()
            if l["author"] == author
        ]
        
        total_calls = sum(l["total_calls"] for l in author_listings)
        total_earnings = sum(l["total_earnings"] for l in author_listings)
        active_listings = sum(1 for l in author_listings if l["active"])
        
        return {
            "author": author,
            "total_listings": len(author_listings),
            "active_listings": active_listings,
            "total_calls": total_calls,
            "total_earnings": round(total_earnings, 2),
        }
    
    def get_platform_stats(self) -> Dict:
        """
        Статистика платформы.
        """
        total_calls = sum(l["total_calls"] for l in self.listings.values())
        total_author_earnings = sum(l["total_earnings"] for l in self.listings.values())
        total_platform_fees = sum(
            t["platform_fee"] for t in self.transactions
        )
        
        tasks = {}
        for l in self.listings.values():
            task = l["task"]
            tasks[task] = tasks.get(task, 0) + 1
        
        return {
            "total_listings": len(self.listings),
            "active_listings": sum(1 for l in self.listings.values() if l["active"]),
            "total_calls": total_calls,
            "total_author_earnings": round(total_author_earnings, 2),
            "total_platform_fees": round(total_platform_fees, 2),
            "total_transactions": len(self.transactions),
            "by_task": tasks,
        }
    
    def deactivate_listing(self, listing_id: str):
        """Деактивация листинга."""
        if listing_id not in self.listings:
            raise KeyError(f"Listing {listing_id} not found")
        self.listings[listing_id]["active"] = False
        self._save(self.listings, self.listings_path)
