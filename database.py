from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy.orm import sessionmaker

# Define your database connection
DATABASE_URL = "postgresql://postgres:niko@localhost/meal_db"

# Set up SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Existing model: User preferences (for meal recommendation)
class UserPreference(Base):
    __tablename__ = 'user_preferences'
    id = Column(Integer, primary_key=True, index=True)
    dietary_preference = Column(String, index=True)
    previous_choice = Column(String)

# 🆕 New model: User accounts
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

# Create tables (if not already present)
Base.metadata.create_all(bind=engine)
