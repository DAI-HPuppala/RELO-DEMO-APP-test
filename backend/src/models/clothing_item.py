"""ClothingItem data model."""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
import uuid


class ClothingItem(BaseModel):
    """Represents a returned clothing item being classified."""
    
    # Identification
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    
    # Classification Attributes
    type: Optional[str] = None  # shirt, pants, dress, jacket, etc.
    color_primary: Optional[str] = None
    color_secondary: Optional[str] = None
    pattern: Optional[str] = None  # solid, striped, checkered, floral, etc.
    
    # Physical Attributes
    neckline: Optional[str] = None  # v-neck, round, crew, etc.
    sleeves: Optional[str] = None  # long, short, sleeveless, etc.
    closure: Optional[str] = None  # buttons, zip, buckle, etc.
    
    # Brand & Size
    brand: Optional[str] = None
    size: Optional[str] = None
    
    # Condition
    has_damage: bool = False
    damage_type: Optional[str] = None  # tear, stain, fade, hole, etc.
    damage_locations: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    final_confidence: float = 0.0
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
    
    def to_export_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            "item_id": self.item_id,
            "type": self.type,
            "color_primary": self.color_primary,
            "color_secondary": self.color_secondary,
            "pattern": self.pattern,
            "neckline": self.neckline,
            "sleeves": self.sleeves,
            "closure": self.closure,
            "brand": self.brand,
            "size": self.size,
            "has_damage": self.has_damage,
            "damage_type": self.damage_type,
            "damage_locations": self.damage_locations,
            "confidence": self.final_confidence
        }