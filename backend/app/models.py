from pydantic import BaseModel
from typing import List, Optional

class Product(BaseModel):
    id: str
    title: str
    description: str
    price: float
    category: str
    image: Optional[str] = None

class SearchResponseItem(BaseModel):
    score: float
    product: Product
    source: Optional[str] = "semantic"   # 'semantic' or 'textual'

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResponseItem]

