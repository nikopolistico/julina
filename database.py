from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:niko@localhost/meal_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class UserPreference(Base):
    __tablename__ = 'user_preferences'
    id = Column(Integer, primary_key=True, index=True)
    dietary_preference = Column(String, index=True)
    previous_choice = Column(String)

Base.metadata.create_all(bind=engine)
