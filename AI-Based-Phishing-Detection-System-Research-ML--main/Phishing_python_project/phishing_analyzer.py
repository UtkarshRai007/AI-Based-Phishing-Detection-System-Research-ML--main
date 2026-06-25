import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import tldextract
import requests
from datetime import datetime
import socket
import whois
import ssl
import difflib

class URLAnalyzer:
    def __init__(self):
        self.load_phishing_dataset()
        # Expanded dictionary of common brands and their official URLs
        self.common_brands = {
            'google': 'https://www.google.com',
            'gmail': 'https://mail.google.com',
            'youtube': 'https://www.youtube.com',
            'facebook': 'https://www.facebook.com',
            'amazon': 'https://www.amazon.com',
            'amazon.in': 'https://www.amazon.in',
            'amazon.co.uk': 'https://www.amazon.co.uk',
            'amazon.ca': 'https://www.amazon.ca',
            'amazon.de': 'https://www.amazon.de',
            'amazon.fr': 'https://www.amazon.fr',
            'amazon.jp': 'https://www.amazon.co.jp',
            'apple': 'https://www.apple.com',
            'microsoft': 'https://www.microsoft.com',
            'paypal': 'https://www.paypal.com',
            'netflix': 'https://www.netflix.com',
            'instagram': 'https://www.instagram.com',
            'twitter': 'https://twitter.com',
            'linkedin': 'https://www.linkedin.com',
            'whatsapp': 'https://www.whatsapp.com',
            'yahoo': 'https://www.yahoo.com',
            'outlook': 'https://outlook.live.com',
            'bank': None,  # Generic, need specific bank name
            'chase': 'https://www.chase.com',
            'wellsfargo': 'https://www.wellsfargo.com',
            'citibank': 'https://www.citibank.com',
            'bankofamerica': 'https://www.bankofamerica.com',
            'hsbc': 'https://www.hsbc.com',
            'steam': 'https://store.steampowered.com',
            'ebay': 'https://www.ebay.com',
            'walmart': 'https://www.walmart.com',
            'target': 'https://www.target.com',
            'dropbox': 'https://www.dropbox.com',
            'github': 'https://github.com',
            'spotify': 'https://www.spotify.com',
            'airbnb': 'https://www.airbnb.com',
            'uber': 'https://www.uber.com',
            'aol': 'https://www.aol.com',
            'icloud': 'https://www.icloud.com',
        }
        
        # Common misspellings and typosquatting patterns
        self.typo_patterns = {
            'google': ['googl', 'gogle', 'googlle', 'g00gle', 'g0ogle', 'gooogle', 'googel'],
            'facebook': ['faceb00k', 'faceboook', 'facebock', 'facebok', 'facebuk', 'facbook'],
            'amazon': ['amason', 'amazan', 'amazen', 'amazn', 'amaxon', 'ammazon'],
            'paypal': ['payp4l', 'paypaal', 'paypayl', 'paypol', 'paypall', 'paypa1'],
            'microsoft': ['micorsoft', 'microsft', 'microsof', 'micorosoft', 'microsfot'],
            'netflix': ['netflex', 'netfliks', 'netflx', 'netfix', 'netfl1x'],
            'instagram': ['instagam', 'instagrm', 'instargam', 'instegram'],
            'twitter': ['twiter', 'twtter', 'twitter', 'twiter', 'twwitter'],
            'apple': ['appl', 'aple', 'appel', 'applle', 'appl3'],
            'yahoo': ['yaho', 'yah00', 'yah0o', 'yahho', 'yahooo'],
        }
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = [
            'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'ru', 'info', 
            'click', 'link', 'bid', 'party', 'webcam', 'win'
        ]
        
        # Common URL shortening services
        self.shortening_services = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd', 'cli.gs', 
            'ow.ly', 'buff.ly', 'rebrand.ly', 'cutt.ly', 'shorturl.at', 
            'u.to', 'tiny.cc'
        ]

    def load_phishing_dataset(self):
        """Load the phishing dataset for known bad URLs"""
        try:
            self.phishing_dataset = pd.read_csv('phishing_dataset.csv')
            self.phishing_urls = set()
            
            # Handle different column naming conventions
            if 'url' in self.phishing_dataset.columns and 'label' in self.phishing_dataset.columns:
                bad_urls = self.phishing_dataset[self.phishing_dataset['label'].str.lower() == 'bad']['url']
                self.phishing_urls = set(url.lower() for url in bad_urls if isinstance(url, str))
            elif 'URL' in self.phishing_dataset.columns and 'Label' in self.phishing_dataset.columns:
                bad_urls = self.phishing_dataset[self.phishing_dataset['Label'].str.lower() == 'bad']['URL']
                self.phishing_urls = set(url.lower() for url in bad_urls if isinstance(url, str))
        except Exception as e:
            print(f"Error loading phishing dataset: {e}")
            self.phishing_dataset = pd.DataFrame()
            self.phishing_urls = set()

    def analyze_url(self, url):
        """Analyze a URL and return phishing risk information"""
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
            
        features = self.extract_extensive_features(url)
        
        # Check against the dataset for known phishing URLs
        if url.lower() in self.phishing_urls:
            features['known_phishing'] = True
            
        # Advanced brand detection
        features['impersonating_brand'] = self.detect_brand_impersonation(features)
        
        risk_score = self.calculate_risk_score(features)
        reasons = self.generate_risk_reasons(features, risk_score)
        safe_url = self.suggest_safe_url(url, features)
        recommendation = self.generate_recommendation(risk_score)
        
        return {
            'url': url,
            'risk_score': risk_score,
            'risk_level': self.get_risk_level(risk_score),
            'reasons': reasons,
            'safe_url': safe_url,
            'recommendation': recommendation,
            'features': features
        }
    
    def extract_extensive_features(self, url):
        """Extract comprehensive features from URL for analysis"""
        try:
            parsed = urlparse(url)
            extracted = tldextract.extract(url)
            
            # Basic URL structure features
            features = {
                'url': url,
                'url_length': len(url),
                'domain': extracted.domain,
                'subdomain': extracted.subdomain,
                'tld': extracted.suffix,
                'scheme': parsed.scheme,
                'path': parsed.path,
                'query_string': parsed.query,
                'fragment': parsed.fragment,
                'has_www': 1 if extracted.subdomain == 'www' else 0,
                'hostname': parsed.netloc,
                'full_domain': f"{extracted.domain}.{extracted.suffix}",
                'known_phishing': False,  # Default value
            }
            
            # URL characteristics
            features.update({
                'has_https': 1 if parsed.scheme == 'https' else 0,
                'num_dots': url.count('.'),
                'num_digits': sum(c.isdigit() for c in url),
                'num_special_chars': sum(not c.isalnum() and not c.isspace() for c in url),
                'has_at': 1 if '@' in url else 0,
                'has_dash': 1 if '-' in parsed.netloc else 0,
                'has_underscore': 1 if '_' in parsed.netloc else 0,
                'has_double_slash_redirect': 1 if '//' in parsed.path else 0,
                'has_ip': 1 if bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url)) else 0,
                'path_length': len(parsed.path),
                'domain_length': len(extracted.domain),
                'has_suspicious_tld': 1 if extracted.suffix in self.suspicious_tlds else 0,
                'num_subdomains': len(extracted.subdomain.split('.')) if extracted.subdomain else 0,
                'query_length': len(parsed.query),
                'domain_with_digits': 1 if any(c.isdigit() for c in extracted.domain) else 0,
                'domain_with_hexchars': 1 if bool(re.search(r'0x[0-9a-fA-F]+', url)) else 0,
                'uses_shortening_service': 1 if any(domain in url.lower() for domain in self.shortening_services) else 0,
                'abnormal_subdomain': 1 if self.has_abnormal_subdomain(extracted) else 0,
                'contains_brand_terms': 0,  # Will be set in brand detection
                'detected_brands': [],  # Will be set in brand detection
                'is_typosquatting': 0,  # Will be set in brand detection
            })
            
            # Detect brands in URL
            brands, is_typosquatting = self.detect_brands(url, features)
            features['detected_brands'] = brands
            features['is_typosquatting'] = is_typosquatting
            features['contains_brand_terms'] = 1 if brands else 0
            
            # Try to get more details if possible
            try:
                # Check SSL Certificate
                features['has_valid_ssl'] = self.check_ssl_certificate(url) if parsed.scheme == 'https' else 0
            except:
                features['has_valid_ssl'] = 0
                
            try:
                # Website existence/age
                features['domain_age'] = self.get_domain_age(f"{extracted.domain}.{extracted.suffix}")
            except:
                features['domain_age'] = -1  # Unknown
            
            # Get IP address
            try:
                features['ip_address'] = self.get_ip_address(parsed.netloc)
            except:
                features['ip_address'] = 'Unknown'
                
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return basic features if extraction fails
            return {
                'url': url,
                'url_length': len(url),
                'has_https': 1 if url.startswith('https://') else 0,
                'error': str(e),
                'known_phishing': False
            }

    def has_abnormal_subdomain(self, extracted):
        """Check if subdomain pattern is abnormal or suspicious"""
        if not extracted.subdomain:
            return 0
        # Check for unusually long subdomains
        if len(extracted.subdomain) > 20:
            return 1
        # Check for excessive dots in subdomain
        if extracted.subdomain.count('.') > 3:
            return 1
        # Check for numeric-only subdomains
        if extracted.subdomain.isdigit():
            return 1
        # Check for brand terms in subdomain
        for brand in self.common_brands:
            if brand in extracted.domain.lower() and brand in extracted.subdomain.lower():
                if brand != extracted.subdomain.lower():
                    return 1
        return 0

    def detect_brands(self, url, features):
        """Detect brand names in URL with improved accuracy for typosquatting"""
        url_lower = url.lower()
        domain = features.get('domain', '').lower()
        hostname = features.get('hostname', '').lower()
        
        found_brands = []
        is_typosquatting = 0
        
        # Check for exact brand matches
        for brand in self.common_brands:
            brand_lower = brand.lower()
            # Skip domains that are part of brand names (like .co.uk in amazon.co.uk)
            if '.' in brand_lower:
                brand_name = brand_lower.split('.')[0]
                if brand_name == domain:
                    found_brands.append(brand_name)
                    continue
            
            # Direct match
            if brand_lower == domain:
                found_brands.append(brand_lower)
        
        # Advanced typosquatting detection
        if not found_brands:  # Only if no direct matches
            for brand, typos in self.typo_patterns.items():
                if any(typo == domain for typo in typos):
                    found_brands.append(brand)
                    is_typosquatting = 1
            
            # Check for close string similarity with any brand
            for brand in self.common_brands:
                brand_name = brand.split('.')[0] if '.' in brand else brand
                
                # String similarity (using different methods for better accuracy)
                if self.levenshtein_distance(brand_name, domain) <= 2 and brand_name not in found_brands:
                    found_brands.append(brand_name)
                    is_typosquatting = 1
                # Sequence matcher (more sophisticated similarity)
                elif difflib.SequenceMatcher(None, brand_name, domain).ratio() > 0.85 and brand_name not in found_brands:
                    found_brands.append(brand_name)
                    is_typosquatting = 1
        
        # If brands found but not marked as typosquatting yet, check if it's official domain
        if found_brands and not is_typosquatting:
            for brand in found_brands:
                if not self.is_official_domain(url, brand):
                    is_typosquatting = 1
                    break
        
        return found_brands, is_typosquatting

    def is_official_domain(self, url, brand):
        """Check if the URL belongs to the official domain of a brand"""
        if brand not in self.common_brands or not self.common_brands[brand]:
            return False
            
        official_url = self.common_brands[brand]
        extracted_official = tldextract.extract(official_url)
        extracted_current = tldextract.extract(url)
        
        official_domain = f"{extracted_official.domain}.{extracted_official.suffix}"
        current_domain = f"{extracted_current.domain}.{extracted_current.suffix}"
        
        return official_domain == current_domain

    def detect_brand_impersonation(self, features):
        """Detect if URL is impersonating a known brand"""
        if not features.get('detected_brands'):
            return None
            
        # If URL is flagged as typosquatting and we have brands detected
        if features.get('is_typosquatting') == 1:
            return features['detected_brands'][0]  # Return the first detected brand
            
        # Check each detected brand for possible impersonation signs
        for brand in features['detected_brands']:
            # Official domain check
            if not self.is_official_domain(features['url'], brand):
                # Additional checks for impersonation signs
                if any([
                    features.get('has_suspicious_tld') == 1,
                    features.get('domain_with_digits') == 1,
                    features.get('has_dash') == 1,
                    features.get('has_ip') == 1,
                    features.get('abnormal_subdomain') == 1,
                    features.get('domain_age', 365) < 180,  # Domain less than 6 months old
                ]):
                    return brand
                    
        return None

    def levenshtein_distance(self, s1, s2):
        """Calculate the Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def check_ssl_certificate(self, url):
        """Check if URL has valid SSL certificate"""
        try:
            hostname = urlparse(url).netloc
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=3) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    if cert and 'notAfter' in cert:
                        return 1
            return 0
        except:
            return 0
    
    def get_domain_age(self, domain):
        """Get domain age in days"""
        try:
            w = whois.whois(domain)
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation_date = w.creation_date[0]
                else:
                    creation_date = w.creation_date
                    
                domain_age = (datetime.now() - creation_date).days
                return domain_age
            return -1
        except:
            return -1

    def calculate_risk_score(self, features):
        """Calculate phishing risk score from 0-100"""
        score = 0
        
        # Known phishing URL - highest risk
        if features.get('known_phishing', False):
            return 100  # Immediately return highest risk score
        
        # Brand impersonation - very high risk
        if features.get('impersonating_brand'):
            score += 70
            
        # Typosquatting or brand impersonation - major red flag
        if features.get('is_typosquatting', 0) == 1:
            score += 60
        
        # Other risk factors (URL length, HTTPS, special chars, etc.)
        if features.get('url_length', 0) > 100:
            score += 15
        elif features.get('url_length', 0) > 75:
            score += 10
        elif features.get('url_length', 0) > 50:
            score += 5
            
        # HTTPS
        if features.get('has_https', 0) == 0:
            score += 20
            
        # Special characters
        if features.get('has_at', 0) == 1:
            score += 25  # @ symbol is a strong phishing indicator
        if features.get('has_dash', 0) == 1 and features.get('domain_with_digits', 0) == 1:
            score += 20
            
        # IP address in URL - major red flag
        if features.get('has_ip', 0) == 1:
            score += 30
            
        # Suspicious TLD
        if features.get('has_suspicious_tld', 0) == 1:
            score += 25
            
        # URL shorteners - often used to hide phishing URLs
        if features.get('uses_shortening_service', 0) == 1:
            score += 25
            
        # Multiple subdomains
        if features.get('num_subdomains', 0) > 2:
            score += 20
            
        # Abnormal subdomain
        if features.get('abnormal_subdomain', 0) == 1:
            score += 25
            
        # SSL certificate
        if features.get('has_valid_ssl', 1) == 0 and features.get('scheme') == 'https':
            score += 25
            
        # Domain age
        domain_age = features.get('domain_age', -1)
        if domain_age != -1:  # Only consider if we have domain age
            if domain_age < 30:  # Less than 30 days old
                score += 30
            elif domain_age < 180:  # Less than 6 months old
                score += 20
            elif domain_age < 365:  # Less than 1 year old
                score += 10
                
        # Double slash in path (potential redirect)
        if features.get('has_double_slash_redirect', 0) == 1:
            score += 15
            
        # Check for excess of numbers or special characters in domain
        if features.get('domain_length', 0) > 0:
            domain = features.get('domain', '')
            digit_ratio = sum(c.isdigit() for c in domain) / features['domain_length'] if features['domain_length'] > 0 else 0
            if digit_ratio > 0.3:  # More than 30% digits
                score += 20
                
        # Clamp score between 0-100
        return max(0, min(100, score))
    
    def get_risk_level(self, score):
        """Convert risk score to risk level"""
        if score >= 75:
            return "Critical Risk"
        elif score >= 50:
            return "High Risk"
        elif score >= 30:
            return "Medium Risk"
        elif score >= 15:
            return "Low Risk"
        else:
            return "Minimal Risk"
    
    def generate_risk_reasons(self, features, risk_score):
        """Generate reasons for risk assessment"""
        reasons = []
        
        # Known phishing URL
        if features.get('known_phishing', False):
            reasons.append("URL is present in our database of known phishing sites.")
            return reasons  # Stop here if it's a known phishing URL
        
        # If safe, say so
        if risk_score < 15:
            reasons.append("URL appears to be safe.")
            return reasons
            
        # Brand impersonation
        if features.get('impersonating_brand'):
            brand = features.get('impersonating_brand')
            reasons.append(f"URL appears to be impersonating {brand.title()}.")
            
        # For typosquatting
        if features.get('is_typosquatting', 0) == 1 and features.get('detected_brands'):
            brand_names = ", ".join(brand.title() for brand in features['detected_brands'])
            reasons.append(f"URL appears to be mimicking {brand_names} but is not the official website.")
            
        # HTTPS
        if not features.get('has_https', 0):
            reasons.append("URL does not use HTTPS secure connection.")
            
        # IP address
        if features.get('has_ip', 0):
            reasons.append("URL contains an IP address instead of a domain name, which is suspicious.")
            
        # @ symbol
        if features.get('has_at', 0):
            reasons.append("URL contains '@' symbol which can be used to mislead users about the actual destination.")
            
        # Suspicious TLD
        if features.get('has_suspicious_tld', 0):
            reasons.append(f"URL uses a suspicious top-level domain (.{features.get('tld', '')}) commonly associated with phishing.")
            
        # URL shortener
        if features.get('uses_shortening_service', 0):
            reasons.append("URL uses a link shortening service which masks the actual destination.")
            
        # Multiple subdomains
        if features.get('num_subdomains', 0) > 2:
            reasons.append("URL contains multiple subdomains which is uncommon for legitimate sites.")
            
        # Domain with digits and dashes
        if features.get('domain_with_digits', 0) and features.get('has_dash', 0):
            reasons.append("URL contains both digits and dashes in domain name, often used in phishing domains.")
            
        # SSL issues
        if features.get('has_valid_ssl', 1) == 0 and features.get('scheme') == 'https':
            reasons.append("URL claims to be secure (https) but has an invalid SSL certificate.")
            
        # New domain
        domain_age = features.get('domain_age', -1)
        if domain_age != -1 and domain_age < 30:
            reasons.append("Domain was registered very recently (less than 30 days ago), which is suspicious.")
        elif domain_age != -1 and domain_age < 180:
            reasons.append("Domain is relatively new (less than 6 months old).")
            
        # If no specific reasons were identified but risk score is high
        if not reasons and risk_score > 50:
            reasons.append("URL contains unusual patterns typical of phishing sites.")
            
        return reasons
    
    def suggest_safe_url(self, url, features):
        """Suggest a safe alternative URL if applicable"""
        # If we've identified a brand being impersonated
        impersonated_brand = features.get('impersonating_brand')
        if impersonated_brand and impersonated_brand in self.common_brands:
            return self.common_brands[impersonated_brand]
            
        # Check detected brands for typosquatting
        if 'detected_brands' in features and features['detected_brands'] and features.get('is_typosquatting', 0) == 1:
            for brand in features['detected_brands']:
                if brand in self.common_brands and self.common_brands[brand]:
                    return self.common_brands[brand]
        
        # Check detected brands even if not marked as typosquatting
        if 'detected_brands' in features and features['detected_brands']:
            for brand in features['detected_brands']:
                # Handle special cases for Amazon based on TLD or location hints
                if brand == 'amazon':
                    tld = features.get('tld', '').lower()
                    if tld == 'in' or '.in' in features.get('url', '').lower():
                        return 'https://www.amazon.in'
                    elif tld == 'uk' or 'co.uk' in features.get('url', '').lower():
                        return 'https://www.amazon.co.uk'
                    
                # Return the standard URL for the brand
                if brand in self.common_brands and self.common_brands[brand]:
                    return self.common_brands[brand]
        
        # Try to infer brand from domain using string similarity
        domain = features.get('domain', '').lower()
        if domain:
            closest_brand = None
            highest_similarity = 0.7  # Threshold for similarity
            
            for brand, official_url in self.common_brands.items():
                if not official_url:
                    continue
                    
                brand_name = brand.split('.')[0] if '.' in brand else brand
                similarity = difflib.SequenceMatcher(None, brand_name, domain).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_brand = brand
            
            if closest_brand:
                return self.common_brands[closest_brand]
                
        return None
    
    def generate_recommendation(self, risk_score):
        """Generate recommendation based on risk score"""
        if risk_score >= 75:
            return "❌ DO NOT PROCEED. This URL is highly likely to be a phishing site designed to steal your information."
        elif risk_score >= 50:
            return "⚠️ AVOID THIS URL. This URL shows strong indicators of being a phishing or malicious site."
        elif risk_score >= 30:
            return "⚠️ PROCEED WITH CAUTION. This URL has some suspicious characteristics. Verify it's legitimate before providing any information."
        elif risk_score >= 15:
            return "🔍 CONSIDER VERIFYING. This URL has minor suspicious traits. Consider verifying its legitimacy if it's an important service."
        else:
            return "✅ LIKELY SAFE. This URL appears to be legitimate, but always be careful when sharing sensitive information online."

    def get_ip_address(self, hostname):
        """Get the IP address of a hostname"""
        try:
            # Remove port number if present
            if ':' in hostname:
                hostname = hostname.split(':')[0]
            
            # Skip if it's already an IP address
            if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', hostname):
                return hostname
            
            # Resolve hostname to IP address
            ip_address = socket.gethostbyname(hostname)
            return ip_address
        except socket.gaierror:
            return "Unable to resolve"
        except Exception as e:
            return f"Error: {str(e)}"
