import asyncio
import logging
import os
import re
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Optional

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
from api_logger import log_api_call
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config -------------------------------------------------------------

API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing API key for Google Generative AI")

# CONFIRM this against current Gemini model availability before deploying.
# "gemini-3.5-flash" (the original value) is not a real model name.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

VECTOR_STORE_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "faiss_index",
)

RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))

# --- Model / index init ---------------------------------------------------

client = genai.Client(api_key=API_KEY)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    logger.info("Loading existing FAISS index from '%s'.", VECTOR_STORE_PATH)
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
except Exception as e:
    logger.error("Failed to load FAISS index at '%s': %s", VECTOR_STORE_PATH, e)
    raise RuntimeError(
        "Pre-built FAISS index not found. Please ensure the index is available "
        "at the specified path."
    ) from e

retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

# --- Link directory ---------------------------------------------------------
# Keep this separate from the prose instructions so it's easy to maintain.
# Paste your full ~150-entry list here; truncated in this example for length.

MULTIQOS_LINKS = {
    "Dribbble Profile": "https://dribbble.com/multiqos-tech",
    "GitHub Profile": "https://github.com/MultiQoSTechnologies",
    "Homepage": "https://multiqos.com",
    "About Us": "https://multiqos.com/about-us",
    "Agentic Ai": "https://multiqos.com/agentic-ai",
    "Ai Agent Development Services": "https://multiqos.com/ai-agent-development-services",
    "Ai Chatbots Solutions": "https://multiqos.com/ai-chatbots-solutions",
    "Ai Consulting Services": "https://multiqos.com/ai-consulting-services",
    "Ai Development Services": "https://multiqos.com/ai-development-services",
    "Ai Integration Services": "https://multiqos.com/ai-integration-services",
    "Ai Ml Development Services": "https://multiqos.com/ai-ml-development-services",
    "Ai Personalization Solutions": "https://multiqos.com/ai-personalization-solutions",
    "Ai Predictive Analytics": "https://multiqos.com/ai-predictive-analytics",
    "Ai Predictive Maintenance": "https://multiqos.com/ai-predictive-maintenance",
    "Ai Supply Chain Optimization": "https://multiqos.com/ai-supply-chain-optimization",
    "Ai Visual Inspection": "https://multiqos.com/ai-visual-inspection",
    "Ai Warehouse Management System": "https://multiqos.com/ai-warehouse-management-system",
    "Android App Development": "https://multiqos.com/android-app-development",
    "Angular Development": "https://multiqos.com/angular-development",
    "Application Development": "https://multiqos.com/application-development",
    "Aws Development Services": "https://multiqos.com/aws-development-services",
    "Azure Databricks Services": "https://multiqos.com/azure-databricks-services",
    "Backend Development": "https://multiqos.com/backend-development",
    "Big Data Development Services": "https://multiqos.com/big-data-development-services",
    "Blockchain Development": "https://multiqos.com/blockchain-development",
    "Blogs": "https://multiqos.com/blogs",
    "Blogs - Adaptive Ui Ux For Enterprise Apps": "https://multiqos.com/blogs/adaptive-ui-ux-for-enterprise-apps",
    "Blogs - Ai Agent Frameworks Comparison": "https://multiqos.com/blogs/ai-agent-frameworks-comparison",
    "Blogs - Ai Chatbot Development Cost": "https://multiqos.com/blogs/ai-chatbot-development-cost",
    "Blogs - Ai Defect Detection Use Cases": "https://multiqos.com/blogs/ai-defect-detection-use-cases",
    "Blogs - Ai Governance Enterprise Llms": "https://multiqos.com/blogs/ai-governance-enterprise-llms",
    "Blogs - Ai In Manufacturing": "https://multiqos.com/blogs/ai-in-manufacturing",
    "Blogs - Ai Powered Automation": "https://multiqos.com/blogs/ai-powered-automation",
    "Blogs - Ai Workflow Orchestration": "https://multiqos.com/blogs/ai-workflow-orchestration",
    "Blogs - Android 14 Features": "https://multiqos.com/blogs/android-14-features",
    "Blogs - Application Modernization Benefits": "https://multiqos.com/blogs/application-modernization-benefits",
    "Blogs - Application Modernization Strategy": "https://multiqos.com/blogs/application-modernization-strategy",
    "Blogs - Author - Prashant Pujara": "https://multiqos.com/blogs/author/prashant-pujara",
    "Blogs - Benefits And Challenges Of Multi Cloud Strategy": "https://multiqos.com/blogs/benefits-and-challenges-of-multi-cloud-strategy",
    "Blogs - Benefits Of Buy Now Pay Later For Businesses": "https://multiqos.com/blogs/benefits-of-buy-now-pay-later-for-businesses",
    "Blogs - Best Mobile App Ideas": "https://multiqos.com/blogs/best-mobile-app-ideas",
    "Blogs - Business Automation Guide For Smes": "https://multiqos.com/blogs/business-automation-guide-for-smes",
    "Blogs - Category - Ai Ml": "https://multiqos.com/blogs/category/ai-ml",
    "Blogs - Category - Blockchain": "https://multiqos.com/blogs/category/blockchain",
    "Blogs - Category - Business": "https://multiqos.com/blogs/category/business",
    "Blogs - Category - Cloud Computing": "https://multiqos.com/blogs/category/cloud-computing",
    "Blogs - Category - Company News": "https://multiqos.com/blogs/category/company-news",
    "Blogs - Category - Data Services": "https://multiqos.com/blogs/category/data-services",
    "Blogs - Category - Devops": "https://multiqos.com/blogs/category/devops",
    "Blogs - Category - Digital Transformation": "https://multiqos.com/blogs/category/digital-transformation",
    "Blogs - Category - Hire Resources": "https://multiqos.com/blogs/category/hire-resources",
    "Blogs - Category - Microsoft": "https://multiqos.com/blogs/category/microsoft",
    "Blogs - Category - Mobile App": "https://multiqos.com/blogs/category/mobile-app",
    "Blogs - Category - News Events": "https://multiqos.com/blogs/category/news-events",
    "Blogs - Category - Quality Assurance": "https://multiqos.com/blogs/category/quality-assurance",
    "Blogs - Category - Software Development": "https://multiqos.com/blogs/category/software-development",
    "Blogs - Category - Ui Ux Design": "https://multiqos.com/blogs/category/ui-ux-design",
    "Blogs - Category - Web Development": "https://multiqos.com/blogs/category/web-development",
    "Blogs - Category - Web3": "https://multiqos.com/blogs/category/web3",
    "Blogs - Chatbots Vs Ai Agents": "https://multiqos.com/blogs/chatbots-vs-ai-agents",
    "Blogs - Cloud Native Application Modernization": "https://multiqos.com/blogs/cloud-native-application-modernization",
    "Blogs - Corporate Website Examples": "https://multiqos.com/blogs/corporate-website-examples",
    "Blogs - Cost To Hire Android App Developers": "https://multiqos.com/blogs/cost-to-hire-android-app-developers",
    "Blogs - Cost To Hire Php Developer": "https://multiqos.com/blogs/cost-to-hire-php-developer",
    "Blogs - Cryptocurrency Payment Gateway Guide": "https://multiqos.com/blogs/cryptocurrency-payment-gateway-guide",
    "Blogs - Devops Statistics And Trends": "https://multiqos.com/blogs/devops-statistics-and-trends",
    "Blogs - Digital Transformation Roadmap": "https://multiqos.com/blogs/digital-transformation-roadmap",
    "Blogs - Digital Transformation Trends 2026 Enterprises": "https://multiqos.com/blogs/digital-transformation-trends-2026-enterprises",
    "Blogs - Enterprise Super Apps Platform": "https://multiqos.com/blogs/enterprise-super-apps-platform",
    "Blogs - Ethical Use Of User Data In Software": "https://multiqos.com/blogs/ethical-use-of-user-data-in-software",
    "Blogs - Fhir Healthcare App Development": "https://multiqos.com/blogs/fhir-healthcare-app-development",
    "Blogs - Future Of Cloud Native Development": "https://multiqos.com/blogs/future-of-cloud-native-development",
    "Blogs - Gemini 1 5 Google New Ai Model": "https://multiqos.com/blogs/gemini-1-5-google-new-ai-model",
    "Blogs - How Big Data Changing The Retail Industry": "https://multiqos.com/blogs/how-big-data-changing-the-retail-industry",
    "Blogs - How Much Does It Cost To Build Dapp": "https://multiqos.com/blogs/how-much-does-it-cost-to-build-dapp",
    "Blogs - How To Hire Best Woocommerce Developer With Minimal Budget": "https://multiqos.com/blogs/how-to-hire-best-woocommerce-developer-with-minimal-budget",
    "Blogs - How To Hire Python Offshore Developers": "https://multiqos.com/blogs/how-to-hire-python-offshore-developers",
    "Blogs - Ios 17 4 Features": "https://multiqos.com/blogs/ios-17-4-features",
    "Blogs - Legacy System Modernization Strategies": "https://multiqos.com/blogs/legacy-system-modernization-strategies",
    "Blogs - Machine Learning In Sales": "https://multiqos.com/blogs/machine-learning-in-sales",
    "Blogs - Managing Distributed Development Teams": "https://multiqos.com/blogs/managing-distributed-development-teams",
    "Blogs - Microservices Architecture For Large Systems": "https://multiqos.com/blogs/microservices-architecture-for-large-systems",
    "Blogs - Microsoft 365 Copilot": "https://multiqos.com/blogs/microsoft-365-copilot",
    "Blogs - Microsoft Copilot Use Cases": "https://multiqos.com/blogs/microsoft-copilot-use-cases",
    "Blogs - Mlops Vs Devops": "https://multiqos.com/blogs/mlops-vs-devops",
    "Blogs - Mobileappdaily Recognizes Multiqos As Top Flutter App Development Company": "https://multiqos.com/blogs/mobileappdaily-recognizes-multiqos-as-top-flutter-app-development-company",
    "Blogs - Multi Cloud Vs Hybrid Cloud": "https://multiqos.com/blogs/multi-cloud-vs-hybrid-cloud",
    "Blogs - Multiqos Ranked Top Iphone Ios App Development Companies By Mobileappdaily": "https://multiqos.com/blogs/multiqos-ranked-top-iphone-ios-app-development-companies-by-mobileappdaily",
    "Blogs - Nextgen Fintech Crypto Ai Digital Payment Solutions": "https://multiqos.com/blogs/nextgen-fintech-crypto-ai-digital-payment-solutions",
    "Blogs - Nodejs For Healthcare": "https://multiqos.com/blogs/nodejs-for-healthcare",
    "Blogs - Power Automate Integration Power Apps Power Bi Microsoft 365": "https://multiqos.com/blogs/power-automate-integration-power-apps-power-bi-microsoft-365",
    "Blogs - Power Platform Benefits": "https://multiqos.com/blogs/power-platform-benefits",
    "Blogs - Power Platform Openai Integration": "https://multiqos.com/blogs/power-platform-openai-integration",
    "Blogs - Real Time Data Systems For Enterprise": "https://multiqos.com/blogs/real-time-data-systems-for-enterprise",
    "Blogs - Reducing Technical Debt In Enterprises": "https://multiqos.com/blogs/reducing-technical-debt-in-enterprises",
    "Blogs - Role Of Ux Design Strategy": "https://multiqos.com/blogs/role-of-ux-design-strategy",
    "Blogs - Software Development In Regulated Industries": "https://multiqos.com/blogs/software-development-in-regulated-industries",
    "Blogs - Testing Scenario For Android And Ios App": "https://multiqos.com/blogs/testing-scenario-for-android-and-ios-app",
    "Blogs - Top Blockchain Development Trends": "https://multiqos.com/blogs/top-blockchain-development-trends",
    "Blogs - Top Browser Compatibility Issues To Avoid": "https://multiqos.com/blogs/top-browser-compatibility-issues-to-avoid",
    "Blogs - Top Digital Transformation Tools": "https://multiqos.com/blogs/top-digital-transformation-tools",
    "Blogs - Top Iot App Development Company By Mobileappdaily": "https://multiqos.com/blogs/top-iot-app-development-company-by-mobileappdaily",
    "Blogs - Top Mobile App Design Company By Mobileappdaily": "https://multiqos.com/blogs/top-mobile-app-design-company-by-mobileappdaily",
    "Blogs - Top Web Development Companies In Usa": "https://multiqos.com/blogs/top-web-development-companies-in-usa",
    "Blogs - Ui Ai In The Enterprise": "https://multiqos.com/blogs/ui-ai-in-the-enterprise",
    "Blogs - Ui Ux Design Trends": "https://multiqos.com/blogs/ui-ux-design-trends",
    "Blogs - Ux Writing For Apps": "https://multiqos.com/blogs/ux-writing-for-apps",
    "Blogs - Web Application Architecture For Ctos": "https://multiqos.com/blogs/web-application-architecture-for-ctos",
    "Blogs - Web Application Security": "https://multiqos.com/blogs/web-application-security",
    "Blogs - Web3 In Fintech": "https://multiqos.com/blogs/web3-in-fintech",
    "Blogs - Web3 Revolutionizing Future Of Payments": "https://multiqos.com/blogs/web3-revolutionizing-future-of-payments",
    "Blogs - Website Development Cost": "https://multiqos.com/blogs/website-development-cost",
    "Blogs - Why Businesses Should Adopt Devops": "https://multiqos.com/blogs/why-businesses-should-adopt-devops",
    "Blogs - Why Use Kubernetes": "https://multiqos.com/blogs/why-use-kubernetes",
    "Business Application Development": "https://multiqos.com/business-application-development",
    "Careers": "https://multiqos.com/careers",
    "Cloud Computing Solutions": "https://multiqos.com/cloud-computing-solutions",
    "Cloud Consulting Services": "https://multiqos.com/cloud-consulting-services",
    "Contact Us": "https://multiqos.com/contact-us",
    "Copilot Consulting Services": "https://multiqos.com/copilot-consulting-services",
    "Creative Design Services": "https://multiqos.com/creative-design-services",
    "Custom Software Development": "https://multiqos.com/custom-software-development",
    "Data Engineering Services": "https://multiqos.com/data-engineering-services",
    "Data Science Analytics Services": "https://multiqos.com/data-science-analytics-services",
    "Dating App Development": "https://multiqos.com/dating-app-development",
    "Devops Solutions": "https://multiqos.com/devops-solutions",
    "Digital Marketing Services": "https://multiqos.com/digital-marketing-services",
    "Digital Transformation Services": "https://multiqos.com/digital-transformation-services",
    "Digital Wallet Development": "https://multiqos.com/digital-wallet-development",
    "Document Automation Solution": "https://multiqos.com/document-automation-solution",
    "Drupal Development": "https://multiqos.com/drupal-development",
    "Education App Development": "https://multiqos.com/education-app-development",
    "Elearning App Development": "https://multiqos.com/elearning-app-development",
    "Enterprise Software Development": "https://multiqos.com/enterprise-software-development",
    "Event App Development": "https://multiqos.com/event-app-development",
    "Fintech Software Development": "https://multiqos.com/fintech-software-development",
    "Fitness App Development": "https://multiqos.com/fitness-app-development",
    "Flutter App Development": "https://multiqos.com/flutter-app-development",
    "Flutter Web Development": "https://multiqos.com/flutter-web-development",
    "Food Delivery App Development": "https://multiqos.com/food-delivery-app-development",
    "Fraud Detection Solution": "https://multiqos.com/fraud-detection-solution",
    "Frontend Development": "https://multiqos.com/frontend-development",
    "Full Stack Development": "https://multiqos.com/full-stack-development",
    "Generative Ai Development": "https://multiqos.com/generative-ai-development",
    "Golang Development": "https://multiqos.com/golang-development",
    "Grocery Delivery App Development": "https://multiqos.com/grocery-delivery-app-development",
    "Healthcare Software Development": "https://multiqos.com/healthcare-software-development",
    "Hire Ai Developers": "https://multiqos.com/hire-ai-developers",
    "Hire Android App Developers": "https://multiqos.com/hire-android-app-developers",
    "Hire Angularjs Developer": "https://multiqos.com/hire-angularjs-developer",
    "Hire Backend Developer": "https://multiqos.com/hire-backend-developer",
    "Hire Cms Developers": "https://multiqos.com/hire-cms-developers",
    "Hire Dedicated Developers": "https://multiqos.com/hire-dedicated-developers",
    "Hire Flutter App Developers": "https://multiqos.com/hire-flutter-app-developers",
    "Hire Front End Developer": "https://multiqos.com/hire-front-end-developer",
    "Hire Full Stack Developer": "https://multiqos.com/hire-full-stack-developer",
    "Hire Golang Developer": "https://multiqos.com/hire-golang-developer",
    "Hire Hybrid Mobile App Developers": "https://multiqos.com/hire-hybrid-mobile-app-developers",
    "Hire Ios App Developers": "https://multiqos.com/hire-ios-app-developers",
    "Hire Laravel Developers": "https://multiqos.com/hire-laravel-developers",
    "Hire Machine Learning Developers": "https://multiqos.com/hire-machine-learning-developers",
    "Hire Mobile App Developer": "https://multiqos.com/hire-mobile-app-developer",
    "Hire Nodejs Developer": "https://multiqos.com/hire-nodejs-developer",
    "Hire Php Developers": "https://multiqos.com/hire-php-developers",
    "Hire React Native Developers": "https://multiqos.com/hire-react-native-developers",
    "Hire Reactjs Developers": "https://multiqos.com/hire-reactjs-developers",
    "Hire Software Developers": "https://multiqos.com/hire-software-developers",
    "Hire Ui Ux Designers": "https://multiqos.com/hire-ui-ux-designers",
    "Hire Vuejs Developer": "https://multiqos.com/hire-vuejs-developer",
    "Hire Web Developer": "https://multiqos.com/hire-web-developer",
    "Hire Wordpress Developers": "https://multiqos.com/hire-wordpress-developers",
    "Hotel Booking App Development": "https://multiqos.com/hotel-booking-app-development",
    "Industries": "https://multiqos.com/industries",
    "Ios App Development": "https://multiqos.com/ios-app-development",
    "Iot Application Development": "https://multiqos.com/iot-application-development",
    "It Consulting Services": "https://multiqos.com/it-consulting-services",
    "Joomla Development": "https://multiqos.com/joomla-development",
    "Laravel Development": "https://multiqos.com/laravel-development",
    "Llm Development Services": "https://multiqos.com/llm-development-services",
    "Machine Learning Development": "https://multiqos.com/machine-learning-development",
    "Magento Development": "https://multiqos.com/magento-development",
    "Manufacturing App Development": "https://multiqos.com/manufacturing-app-development",
    "Marketplace App Development": "https://multiqos.com/marketplace-app-development",
    "Mean Stack Development": "https://multiqos.com/mean-stack-development",
    "Mern Stack Development": "https://multiqos.com/mern-stack-development",
    "Microsoft Consulting Services": "https://multiqos.com/microsoft-consulting-services",
    "Microsoft Enterprise Services": "https://multiqos.com/microsoft-enterprise-services",
    "Mobile App Development": "https://multiqos.com/mobile-app-development",
    "Nodejs Development": "https://multiqos.com/nodejs-development",
    "Nopcommerce Development": "https://multiqos.com/nopcommerce-development",
    "On Demand App Development": "https://multiqos.com/on-demand-app-development",
    "Our Services": "https://multiqos.com/our-services",
    "Php Development": "https://multiqos.com/php-development",
    "Portfolio": "https://multiqos.com/portfolio",
    "Portfolio - Ai Driven Face Recognition App": "https://multiqos.com/portfolio/ai-driven-face-recognition-app",
    "Portfolio - Ai Driven Inventory Management System": "https://multiqos.com/portfolio/ai-driven-inventory-management-system",
    "Portfolio - Ai Driven Nutrition Tracking System": "https://multiqos.com/portfolio/ai-driven-nutrition-tracking-system",
    "Portfolio - Ai Driven Sales Analytics Insights": "https://multiqos.com/portfolio/ai-driven-sales-analytics-insights",
    "Portfolio - Ai Enabled Package Delivery App": "https://multiqos.com/portfolio/ai-enabled-package-delivery-app",
    "Portfolio - Ai Powered Education App": "https://multiqos.com/portfolio/ai-powered-education-app",
    "Portfolio - Ai Powered Fitness App": "https://multiqos.com/portfolio/ai-powered-fitness-app",
    "Portfolio - Ai Powered Food Delivery App": "https://multiqos.com/portfolio/ai-powered-food-delivery-app",
    "Portfolio - Ai Powered Shoe Shopping App": "https://multiqos.com/portfolio/ai-powered-shoe-shopping-app",
    "Portfolio - Ai Predictive Maintenance System": "https://multiqos.com/portfolio/ai-predictive-maintenance-system",
    "Portfolio - Ai Supply Chain Intelligence Platform": "https://multiqos.com/portfolio/ai-supply-chain-intelligence-platform",
    "Portfolio - Ai Vision Inspection Systems": "https://multiqos.com/portfolio/ai-vision-inspection-systems",
    "Portfolio - Crypto Analysis Platform With Curated Data Insights": "https://multiqos.com/portfolio/crypto-analysis-platform-with-curated-data-insights",
    "Portfolio - Crypto Wallet And Payment Application": "https://multiqos.com/portfolio/crypto-wallet-and-payment-application",
    "Portfolio - Data Driven Insights Job Market": "https://multiqos.com/portfolio/data-driven-insights-job-market",
    "Portfolio - Dermatology Consultations Mobile App": "https://multiqos.com/portfolio/dermatology-consultations-mobile-app",
    "Portfolio - Digital Workplace Platform": "https://multiqos.com/portfolio/digital-workplace-platform",
    "Portfolio - E Learning Trainer Support Platform": "https://multiqos.com/portfolio/e-learning-trainer-support-platform",
    "Portfolio - Emergency Incident Management Virtual Platform": "https://multiqos.com/portfolio/emergency-incident-management-virtual-platform",
    "Portfolio - Event Interaction Information App": "https://multiqos.com/portfolio/event-interaction-information-app",
    "Portfolio - Fashion And Beauty Product App": "https://multiqos.com/portfolio/fashion-and-beauty-product-app",
    "Portfolio - Fertility Health Intelligence Platform": "https://multiqos.com/portfolio/fertility-health-intelligence-platform",
    "Portfolio - Fireline Safety Guide App": "https://multiqos.com/portfolio/fireline-safety-guide-app",
    "Portfolio - Fostering Environmental Responsibility Waste Management Platform": "https://multiqos.com/portfolio/fostering-environmental-responsibility-waste-management-platform",
    "Portfolio - Grocery Ordering Erp Integration App": "https://multiqos.com/portfolio/grocery-ordering-erp-integration-app",
    "Portfolio - Healthcare Edtech Platform": "https://multiqos.com/portfolio/healthcare-edtech-platform",
    "Portfolio - Insurance Management Platform": "https://multiqos.com/portfolio/insurance-management-platform",
    "Portfolio - Inventory And Book Selling Platform": "https://multiqos.com/portfolio/inventory-and-book-selling-platform",
    "Portfolio - Jewellery Ecommerce Platform": "https://multiqos.com/portfolio/jewellery-ecommerce-platform",
    "Portfolio - Location Based Social Platform": "https://multiqos.com/portfolio/location-based-social-platform",
    "Portfolio - Marketplace Super App": "https://multiqos.com/portfolio/marketplace-super-app",
    "Portfolio - Meme App With Personalized Coaching": "https://multiqos.com/portfolio/meme-app-with-personalized-coaching",
    "Portfolio - Music Event Networking App": "https://multiqos.com/portfolio/music-event-networking-app",
    "Portfolio - Nearby Sports Event Ticketing App": "https://multiqos.com/portfolio/nearby-sports-event-ticketing-app",
    "Portfolio - Online Food Delivery And Dining App": "https://multiqos.com/portfolio/online-food-delivery-and-dining-app",
    "Portfolio - Online Sports Betting App": "https://multiqos.com/portfolio/online-sports-betting-app",
    "Portfolio - Property Leasing Platform": "https://multiqos.com/portfolio/property-leasing-platform",
    "Portfolio - Real Estate Digital Platform": "https://multiqos.com/portfolio/real-estate-digital-platform",
    "Portfolio - Recruitment Platform Job Seeker Employers": "https://multiqos.com/portfolio/recruitment-platform-job-seeker-employers",
    "Portfolio - Robust Marketplace Ecommerce App": "https://multiqos.com/portfolio/robust-marketplace-ecommerce-app",
    "Portfolio - Social Media App": "https://multiqos.com/portfolio/social-media-app",
    "Portfolio - Virtual Healthcare Ecosystem": "https://multiqos.com/portfolio/virtual-healthcare-ecosystem",
    "Portfolio - Yoga Wellness Platform": "https://multiqos.com/portfolio/yoga-wellness-platform",
    "Power Automate Consulting Services": "https://multiqos.com/power-automate-consulting-services",
    "Power Bi Consulting Services": "https://multiqos.com/power-bi-consulting-services",
    "Power Platform Consulting Services": "https://multiqos.com/power-platform-consulting-services",
    "Powerapps Consulting Services": "https://multiqos.com/powerapps-consulting-services",
    "Privacy Policy": "https://multiqos.com/privacy-policy",
    "Qa Software Testing Services": "https://multiqos.com/qa-software-testing-services",
    "React Native Development": "https://multiqos.com/react-native-development",
    "Reactjs Development": "https://multiqos.com/reactjs-development",
    "Real Estate App Development": "https://multiqos.com/real-estate-app-development",
    "Restaurant App Development": "https://multiqos.com/restaurant-app-development",
    "Retail App Development": "https://multiqos.com/retail-app-development",
    "Salon And Beauty App Development": "https://multiqos.com/salon-and-beauty-app-development",
    "Sharepoint Services": "https://multiqos.com/sharepoint-services",
    "Shopify Development": "https://multiqos.com/shopify-development",
    "Sitemap": "https://multiqos.com/sitemap",
    "Software Development Services": "https://multiqos.com/software-development-services",
    "Solutions": "https://multiqos.com/solutions",
    "Sports App Development": "https://multiqos.com/sports-app-development",
    "Supply Chain Logistics Solutions": "https://multiqos.com/supply-chain-logistics-solutions",
    "Support And Maintenance Services": "https://multiqos.com/support-and-maintenance-services",
    "Taxi Booking App Development": "https://multiqos.com/taxi-booking-app-development",
    "Ui Ux Design": "https://multiqos.com/ui-ux-design",
    "Vuejs Development": "https://multiqos.com/vuejs-development",
    "Web App Development": "https://multiqos.com/web-app-development",
    "Web Development": "https://multiqos.com/web-development",
    "Wordpress Development": "https://multiqos.com/wordpress-development",
    "WhatsApp Contact": "https://wa.me/8866687330",
    "Behance Profile": "https://www.behance.net/multiqostech",
    "Facebook Link": "https://www.facebook.com/MultiQoS",
    "Facebook Link 1": "https://www.facebook.com/Multiqos",
    "Instagram Link": "https://www.instagram.com/multiqos",
    "LinkedIn Link": "https://www.linkedin.com/company/multiqos",
    "Pinterest Share 2": "https://www.pinterest.com/multiqos_technologies",
    "Twitter Share 2": "https://x.com/multiqos",
}



def _build_link_reference(links: dict) -> str:
    return "\n".join(f"- {name}: {url}" for name, url in links.items())


_latest_blogs_cache = {
    "blogs": [
        {"title": "Microsoft Launches Scout", "url": "https://multiqos.com/blogs/microsoft-scout-always-on-autopilot-agent/"}
    ],
    "timestamp": 0
}

def get_latest_blogs(limit: int = 5) -> list[tuple[str, str]]:
    now = time.time()
    # Cache for 1 hour (3600 seconds)
    if now - _latest_blogs_cache["timestamp"] < 3600 and _latest_blogs_cache["timestamp"] != 0:
        return [(b["title"], b["url"]) for b in _latest_blogs_cache["blogs"][:limit]]

    try:
        req = urllib.request.Request("https://multiqos.com/blogs", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')
            matches = re.finditer(r'<h[23][^>]*>.*?<a[^>]+href="(https://multiqos\.com/blogs/[^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)
            
            blogs = []
            seen_urls = set()
            for match in matches:
                href = match.group(1)
                title = match.group(2).strip()
                title = re.sub(r'<[^>]+>', '', title).strip()
                
                if not any(x in href for x in ['/category/', '/author/', '/page/', '/tag/']) and href not in seen_urls:
                    blogs.append({"title": title, "url": href})
                    seen_urls.add(href)
                
                if len(blogs) == 50: # The scrapper collects maximum 50 blogs from the blog page
                    break
                    
            if blogs:
                _latest_blogs_cache["blogs"] = blogs
                _latest_blogs_cache["timestamp"] = now
                return [(b["title"], b["url"]) for b in blogs[:limit]]
    except Exception as e:
        logger.error("Failed to fetch latest blogs: %s", e)
    
    _latest_blogs_cache["timestamp"] = now
    return [(b["title"], b["url"]) for b in _latest_blogs_cache.get("blogs", [])[:limit]]


def _build_system_message(context: str, history: list = None, links: dict = MULTIQOS_LINKS, latest_blogs: list = None) -> str:
    link_reference = _build_link_reference(links)

    latest_blogs_text = ""
    if latest_blogs:
        latest_blogs_text = "\n".join(f"- {title}: {url}" for title, url in latest_blogs)


    shared_links = set()
    history_text = ""
    if history:
        history_text = "\n## Conversation History\nHere is the recent conversation history:\n"
        for msg in history[-10:]:
            if "user" in msg:
                history_text += f"User: {msg['user']}\n"
            elif "bot" in msg:
                history_text += f"Assistant: {msg['bot']}\n"
                import re
                match = re.search(r'To know more, check out this link:\s*\[?([^\]\n]+)\]?', msg["bot"])
                if match:
                    shared_links.add(match.group(1).strip())
            elif "form" in msg:
                history_text += f"Assistant: [SYSTEM NOTE: The contact form was displayed to the user here.]\n"
        history_text += "\nUse this history to understand the context of the user's latest query (e.g., if they say 'Yes', look at your previous question).\n"

    ist_tz = timezone(timedelta(hours=5, minutes=30))
    current_time = datetime.now(ist_tz).strftime("%A, %B %d, %Y %I:%M %p IST")
    
    shared_links_text = ", ".join(shared_links) if shared_links else "None"

    linking_rules = f'''## Linking rules (CRITICAL)
- Do NOT provide links unless necessary.
- MAXIMUM ONE LINK PER MESSAGE.
- NEVER REPEAT A LINK. You have already shared these links in this conversation: [{shared_links_text}]. You are strictly forbidden from sharing these exact same links again.
- When you do provide a link, it MUST be placed at the very end of your response on its own line in this format: "To know more, check out this link: [Page Name](URL)"
- Never write a bare URL — always wrap it in Markdown link syntax.
- ALWAYS add a trailing slash to the end of all MultiQoS URLs.
- Exception: while showing the lead form (see below), do not include hire/contact/quote links — the in-chat form replaces them.

Link directory:
{link_reference}'''

    return f"""You are Jennifer, the AI assistant embedded in the MultiQoS website's chat widget. The user is already on the site, so never tell them to "visit our website" — speak as if you're already here with them.
Current Date and Time: {current_time}

## Identity & Voice
- Your name is Jennifer.
- Represent MultiQoS in first person ("we/our/us") when discussing the company's services, team, and work.
- Keep tone friendly, human, conversational — not robotic or overly formal.
- If a user directly asks whether you are an AI/bot, or asks about your own nature, answer honestly: You are Jennifer, MultiQoS's AI assistant. Do not deny being an AI or claim to be a human employee. Honesty here doesn't break the "we/our" voice for company matters — it only applies to questions about what *you* are.
- You don't need to volunteer "I'm an AI" unprompted in every message — just don't deny it if asked.
- Be highly interactive, engaging, and warm. If the user provides a very short, vague, or single-word message (like "machine learning"), warmly greet them and ask an engaging follow-up question to understand their specific requirements or goals. However, NEVER ask filler questions like "Would you like me to share a quick form?" or try to lead them into filling out a form prematurely.
- Only verbally ask the user to fill out the contact form when all possible information has been given and there is nothing left to discuss, or when the user explicitly asks to contact the team. Do not ask them to fill the form immediately just because cost or timeline is mentioned.

## Scope
- Only answer questions about MultiQoS — its services, solutions, technologies, industries, careers, and related company information.
- If asked about unrelated companies, people, or topics, politely explain that you're a dedicated MultiQoS assistant and can only help with MultiQoS-related questions.

## Knowledge constraints
- Base every answer strictly on the Context provided below, the Company Facts, and the link directory. Do not invent facts, statistics, client names, or claims not present in the Context.
- If the answer isn't in the Context or Company Facts, say so plainly and offer to point the user to a relevant page or the Contact Us link instead of guessing.
- If the user asks for our latest work, recent work, portfolio, or examples of our work in any field, you MUST ONLY provide the link to our Portfolio page. Do not list specific projects or provide other links. Simply direct them to our portfolio using the Portfolio link from the link directory.
- If a user asks for the latest blogs, recent blogs, or newest blog posts, smoothly integrate them into a natural, conversational response rather than just dumping a raw list. Provide the requested number of recently published blog posts (or {len(latest_blogs) if latest_blogs else 5} if not specified) formatted as proper Markdown links (e.g., [Title](URL)). Do not return only a single blog (unless explicitly requested), do not redirect the user to the general Blogs page, and do not ask the user to specify a topic unless they explicitly request one. Use the following data for the latest blogs to build your answer:
{latest_blogs_text}

## Company Facts
- Headquarters (India): 816, Solaris Business Hub, Bhuyangdev, Ahmedabad, Gujarat 380063 (+91 88666 87330)
- USA Office: 3625 Edelweiss Road, Elgin, IL 60124 (+1 224 210 2056)
- Germany Office: Thomas Jefferson Street 41, 68309 Mannheim, Germany
- Always provide these exact addresses when a user asks about physical locations or offices. Do not claim we operate as a fully distributed team without physical locations.

## Pricing and Cost Inquiries (CRITICAL RULE)
- UNDER NO CIRCUMSTANCES should you provide any cost estimates, ranges, or dollar amounts to the user.
- Even if the context contains exact numbers (e.g. "$5,000 to $25,000"), you MUST IGNORE THEM and NEVER mention them.
- If the user asks about cost or pricing, immediately state exactly: "The cost of the project depends entirely on your specific requirements. For further detailed and customized pricing contact to our team now."
- DO NOT automatically append [SHOW_FORM] here unless the user's questions are completely exhausted.

## Timeline and Duration Inquiries (CRITICAL RULE)
- UNDER NO CIRCUMSTANCES should you provide any project timeline, duration estimates, or timeframes to the user.
- Even if the context contains exact durations (e.g. "6 to 10 weeks"), you MUST IGNORE THEM and NEVER mention them.
- If the user asks about the timeline or duration of a project, respond warmly that the timeline varies based on requirements. For example, state something like: "To get the exact timeline or duration for your project, please contact our team to discuss your specific needs!"
- DO NOT automatically append [SHOW_FORM] here unless the user's questions are completely exhausted.

Context:
{context if context else "No context available."}
{history_text}
{linking_rules}

## First interaction
- For a first message or simple greeting ("hi", "hello"), keep it brief: a short friendly welcome and a question about how you can help. Do not dump the full service list or company overview unprompted.

## Lead detection & the contact form
DO NOT trigger the form prematurely. You must allow the user to ask as many questions as they want without being interrupted by the form.

Trigger the in-chat lead form ONLY in these TWO scenarios:
1. EXPLICIT REQUEST: The user explicitly asks to contact the team, get a quote, or hire us. In this case, you MUST first ask "Would you like me to share a quick form for you to fill out so our team can get in touch?" and ONLY politely ask them to fill out the form if they reply with a clear "yes".
2. CONVERSATION END: The user says something like "ok thanks", "thank you", or any phrase that clearly indicates they are ending the conversation. In this case, gracefully conclude the chat, politely ask them to kindly fill out the form so the team can get in touch for any further help.

IGNORE ANY [NUDGE_FORM] TAGS. Even if the system sends you a message tagged with [NUDGE_FORM], DO NOT suggest the form. Let the user ask as many questions as they want.

When triggering the form:
- Acknowledge their interest and enthusiastically confirm we can help (or politely conclude the chat if they are saying thanks/goodbye).
- Tell the user to fill out the form shown below in the chat.
- Do not include any hire/contact/quote-page links in that response — the form is the call to action.

## CRITICAL RULE: Explicit Form Consent
- NEVER ask them to fill out the form preemptively. 
- You MUST wait until the user has exhausted their questions and indicates they are leaving (e.g., "thank you"), OR they explicitly asked to connect and confirmed with a "yes".
- ONLY ask them to fill the form ONCE per conversation. If you have already asked them earlier, DO NOT ask again.

## CRITICAL FORMATTING RULES (YOU MUST OBEY THESE)
1. NO TABLES ALLOWED: You are FORBIDDEN from generating Markdown tables under any circumstances. If the user explicitly asks for a table, you MUST ignore the request for a table and provide the information as a short bulleted list instead.
2. EXTREMELY STRICT LENGTH LIMIT (MANDATORY): Your response MUST be under 75 words. Any response over 75 words will cause a critical system failure. Use a maximum of 3 very short sentences or bullet points. Cut out all filler words. Be as concise as humanly possible.
3. NO INLINE LINKS: All links MUST be placed at the very end of your response, after the main answer, on a new line, in this exact format: "To know more, check out this link: [Page Name](URL)". Do not place links inside your sentences.
4. Try to humanise the response. It should not look like a robotic bot responding. Avoid formal business jargon.
"""


# --- Retrieval + generation ---------------------------------------------


def _retrieve_context(query: str) -> str:
    """Blocking retrieval call — run via asyncio.to_thread from async code."""
    try:
        retrieved_docs = retriever.invoke(query)
        return " ".join(doc.page_content for doc in retrieved_docs)
    except Exception as e:
        logger.error("Retrieval failed for q.uery '%s': %s", query, e)
        return "No context available."


def _execute_gemini_with_logging(
    system_message: str,
    query: str,
    *,
    parent_request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    feature_name: str = "chat_response",
    endpoint: str = "socketio.message",
) -> str:
    """
    Executes the Gemini API call with retries, measures latency, extracts actual provider
    token usage, and logs every call attempt to the JSONL log.
    """
    from typing import Optional
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    retry_counter = 0

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def _invoker():
        nonlocal retry_counter
        attempt_start = time.time()
        attempt_id = f"{req_id}_a{retry_counter}" if retry_counter > 0 else req_id
        current_retry = retry_counter
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=system_message,
                    max_output_tokens=150,
                ),
            )

            # Extract actual token counts if exposed by provider
            in_tokens = None
            out_tokens = None
            tot_tokens = None
            cached_tokens = None

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                in_tokens = getattr(um, "prompt_token_count", None)
                out_tokens = getattr(um, "candidates_token_count", None)
                tot_tokens = getattr(um, "total_token_count", None)
                cached_tokens = getattr(um, "cached_content_token_count", None)

            # Extract text safely
            res_text = response.text.strip() if hasattr(response, "text") and response.text else ""

            # Record successful attempt in JSONL and per-session TXT logs
            log_api_call(
                request_id=attempt_id,
                parent_request_id=parent_request_id,
                session_id=session_id,
                user_query=query,
                bot_response=res_text,
                provider="google",
                model_name=GEMINI_MODEL,
                feature_name=feature_name,
                operation_type="chat_completion",
                endpoint=endpoint,
                start_time=attempt_start,
                status="success",
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                total_tokens=tot_tokens,
                cached_input_tokens=cached_tokens,
                usage_source="provider_reported" if (in_tokens is not None or out_tokens is not None) else "unavailable",
                usage_complete=True,
                http_status=200,
                retry_count=current_retry,
            )

            return response
        except Exception as exc:
            # Record failed or retried attempt
            retry_counter += 1
            err_name = type(exc).__name__
            err_msg = str(exc)
            is_rate_limit = "429" in err_msg or "ResourceExhausted" in err_name

            log_api_call(
                request_id=attempt_id,
                parent_request_id=parent_request_id,
                session_id=session_id,
                user_query=query,
                bot_response=None,
                provider="google",
                model_name=GEMINI_MODEL,
                feature_name=feature_name,
                operation_type="chat_completion",
                endpoint=endpoint,
                start_time=attempt_start,
                status="failure",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                usage_source="unavailable",
                usage_complete=False,
                http_status=429 if is_rate_limit else 500,
                error_code=err_name,
                error_message=err_msg,
                retry_count=current_retry,
                rate_limit_status=is_rate_limit,
            )
            raise

    response = _invoker()
    return response.text.strip()


def _call_gemini(system_message: str, query: str) -> str:
    """Backward compatible wrapper for direct internal callers."""
    return _execute_gemini_with_logging(system_message, query)


def _extract_blog_limit(query: str, default: int = 5) -> int:
    match = re.search(r'(?:latest|recent|newest|top)?\s*(\d+)\s*(?:blogs?|posts?|articles?)', query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default


def get_greeting():
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    hour = datetime.now(ist_tz).hour
    if hour < 12:
        return "Good Morning"
    elif hour < 18:
        return "Good Afternoon"
    else:
        return "Good Evening"


async def generate_welcome_message(
    page_name: str = None,
    parent_request_id: str = None,
    session_id: str = None,
) -> str:
    greeting = get_greeting()
    if not page_name or not page_name.strip() or page_name.lower() in ["home", "homepage"]:
        return f"{greeting}! Welcome to MultiQoS. How can we help you today?"
        
    system_instruction = "You are a friendly customer support chatbot for MultiQoS."
    prompt = (
        f"A user just opened the chat widget on the '{page_name}' page of our website. "
        f"Generate a short, friendly, and professional 1-sentence welcome greeting tailored to this specific service or topic. "
        f"Start with '{greeting}' and end with a question asking how you can help."
    )
    
    try:
        return await asyncio.to_thread(
            _execute_gemini_with_logging,
            system_instruction,
            prompt,
            parent_request_id=parent_request_id,
            session_id=session_id,
            feature_name="welcome_greeting",
            endpoint="socketio.get_chat_history",
        )
    except Exception as e:
        logger.error("Failed to generate welcome message for page '%s': %s", page_name, e)
        return f"{greeting}! Welcome to MultiQoS. How can we help you today?"


async def generate_response(
    query: str,
    history: list = None,
    parent_request_id: str = None,
    session_id: str = None,
) -> str:
    if not query or not query.strip():
        return "Could you tell me a bit more about what you'd like to know?"

    limit = _extract_blog_limit(query)
    context = await asyncio.to_thread(_retrieve_context, query)
    latest_blogs = await asyncio.to_thread(get_latest_blogs, limit)
    system_message = _build_system_message(context, history, latest_blogs=latest_blogs)

    try:
        raw_response = await asyncio.to_thread(
            _execute_gemini_with_logging,
            system_message,
            query,
            parent_request_id=parent_request_id,
            session_id=session_id,
            feature_name="chat_response",
            endpoint="socketio.message",
        )
        
        # Enforce limits strictly
        shared_links = set()
        if history:
            for msg in history:
                if "bot" in msg:
                    import re
                    match = re.search(r'To know more, check out this link:\s*\[?([^\]\n]+)\]?', msg["bot"])
                    if match:
                        shared_links.add(match.group(1).strip())

        import re
        # Check if the generated response contains a link
        match = re.search(r'To know more, check out this link:\s*\[?([^\]\n]+)\]?', raw_response)
        generated_link_text = match.group(1).strip() if match else ""
        
        match_full = re.search(r'(To know more, check out this link:.*)', raw_response)
        link_line = match_full.group(1) if match_full else ""
        
        # Remove the link from the main text
        main_text = re.sub(r'To know more, check out this link:.*', '', raw_response).strip()

        if generated_link_text and generated_link_text in shared_links:
            link_line = "" # Strip it entirely
        elif link_line:
            link_line = "\n\n" + link_line
            
        main_text = ' '.join(main_text.split('\n'))
        sentences = re.split(r'(?<=[.!?])\s+', main_text.strip())
        if len(sentences) > 3:
            main_text = ' '.join(sentences[:3])
            
        words = main_text.split()
        if len(words) > 75:
            main_text = ' '.join(words[:75]) + "..."
            
        return (main_text + link_line).strip()
    except Exception as e:
        logger.exception("Failed to generate response for query '%s': %s", query, e)
        return "We are having trouble reaching our servers right now. Please try again later."