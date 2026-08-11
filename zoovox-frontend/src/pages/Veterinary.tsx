import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ArrowLeft, Search, MapPin, Phone, Clock, ShoppingBag, AlertCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import animalsBackground from "@/assets/animals-background.jpg";

const Veterinary = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<"hospitals" | "food" | "accessories" | "precautions">("hospitals");

  const handleSearch = () => {
    if (searchQuery.trim()) {
      toast.success(`Searching for ${activeTab} near "${searchQuery}"...`);
    } else {
      toast.error("Please enter a location");
    }
  };

  const hospitals = [
    { name: "PetCare Veterinary Hospital", distance: "0.5 km", phone: "+1 234-567-8900", hours: "24/7" },
    { name: "Animal Health Center", distance: "1.2 km", phone: "+1 234-567-8901", hours: "8 AM - 10 PM" },
    { name: "Emergency Pet Clinic", distance: "2.1 km", phone: "+1 234-567-8902", hours: "24/7" },
  ];

  const foodItems = [
    { name: "Premium Dog Food", brand: "NutriPaw", price: "$45.99" },
    { name: "Organic Cat Food", brand: "WhiskerHealth", price: "$38.50" },
    { name: "Bird Seed Mix", brand: "AvianPlus", price: "$22.99" },
  ];

  const accessories = [
    { name: "Smart Pet Collar", price: "$79.99", category: "Tech" },
    { name: "Comfort Pet Bed", price: "$65.00", category: "Furniture" },
    { name: "Interactive Toy Set", price: "$34.50", category: "Toys" },
  ];

  const precautions = [
    { title: "Regular Checkups", description: "Schedule veterinary visits every 6 months for preventive care" },
    { title: "Vaccination Schedule", description: "Keep your pet's vaccinations up to date to prevent diseases" },
    { title: "Nutrition Guide", description: "Feed appropriate portions based on age, size, and activity level" },
    { title: "Emergency Preparedness", description: "Keep emergency vet contacts and first aid kit ready" },
  ];

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background */}
      <div 
        className="absolute inset-0 bg-cover bg-center opacity-10 animate-float"
        style={{ backgroundImage: `url(${animalsBackground})` }}
      />
      <div className="absolute inset-0 bg-background" />
      
      {/* Orbs */}
      <div className="absolute top-20 left-20 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-glow" />
      <div className="absolute bottom-40 right-40 w-80 h-80 bg-secondary/10 rounded-full blur-3xl animate-glow" style={{ animationDelay: "1.5s" }} />

      {/* Header */}
      <header className="relative z-20 border-b border-border/50 backdrop-blur-lg bg-card/30">
        <div className="container mx-auto px-4 py-4 flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate("/dashboard")}
            className="hover:bg-primary/10"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              Veterinary Care
            </h1>
            <p className="text-xs text-muted-foreground">Find everything for your pet's health</p>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="relative z-10 container mx-auto px-4 py-8">
        {/* Search section */}
        <Card className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-6 mb-8 animate-fade-in">
          <div className="flex gap-4">
            <div className="flex-1">
              <Input
                placeholder="Enter your location or zip code..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="bg-background/50 h-12"
              />
            </div>
            <Button 
              onClick={handleSearch}
              className="bg-gradient-primary hover:opacity-90 h-12 px-8"
            >
              <Search className="w-5 h-5 mr-2" />
              Search
            </Button>
          </div>
        </Card>

        {/* Map placeholder */}
        <Card className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-6 mb-8 animate-scale-in">
          <div className="bg-muted/30 rounded-lg h-64 flex items-center justify-center border-2 border-dashed border-border">
            <div className="text-center">
              <MapPin className="w-12 h-12 mx-auto mb-3 text-primary" />
              <p className="text-muted-foreground">Interactive map will be displayed here</p>
              <p className="text-sm text-muted-foreground/60 mt-1">Showing nearby veterinary services</p>
            </div>
          </div>
        </Card>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-2 animate-fade-in" style={{ animationDelay: "0.1s" }}>
          <Button
            variant={activeTab === "hospitals" ? "default" : "outline"}
            onClick={() => setActiveTab("hospitals")}
            className={activeTab === "hospitals" ? "bg-gradient-primary" : "border-primary/50"}
          >
            <MapPin className="w-4 h-4 mr-2" />
            Hospitals
          </Button>
          <Button
            variant={activeTab === "food" ? "default" : "outline"}
            onClick={() => setActiveTab("food")}
            className={activeTab === "food" ? "bg-gradient-primary" : "border-primary/50"}
          >
            <ShoppingBag className="w-4 h-4 mr-2" />
            Food
          </Button>
          <Button
            variant={activeTab === "accessories" ? "default" : "outline"}
            onClick={() => setActiveTab("accessories")}
            className={activeTab === "accessories" ? "bg-gradient-primary" : "border-primary/50"}
          >
            <ShoppingBag className="w-4 h-4 mr-2" />
            Accessories
          </Button>
          <Button
            variant={activeTab === "precautions" ? "default" : "outline"}
            onClick={() => setActiveTab("precautions")}
            className={activeTab === "precautions" ? "bg-gradient-primary" : "border-primary/50"}
          >
            <AlertCircle className="w-4 h-4 mr-2" />
            Precautions
          </Button>
        </div>

        {/* Content based on active tab */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {activeTab === "hospitals" && hospitals.map((hospital, index) => (
            <Card 
              key={index}
              className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-6 hover:shadow-glow transition-all duration-300 animate-scale-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <h3 className="text-xl font-bold mb-4">{hospital.name}</h3>
              <div className="space-y-3 text-muted-foreground">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4 text-primary" />
                  <span>{hospital.distance} away</span>
                </div>
                <div className="flex items-center gap-2">
                  <Phone className="w-4 h-4 text-primary" />
                  <span>{hospital.phone}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-primary" />
                  <span>{hospital.hours}</span>
                </div>
              </div>
              <Button className="w-full mt-6 bg-gradient-primary hover:opacity-90">
                Get Directions
              </Button>
            </Card>
          ))}

          {activeTab === "food" && foodItems.map((item, index) => (
            <Card 
              key={index}
              className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-6 hover:shadow-glow transition-all duration-300 animate-scale-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <h3 className="text-xl font-bold mb-2">{item.name}</h3>
              <p className="text-muted-foreground mb-4">by {item.brand}</p>
              <p className="text-2xl font-bold text-accent mb-4">{item.price}</p>
              <Button className="w-full bg-accent hover:bg-accent/90 text-accent-foreground">
                View Details
              </Button>
            </Card>
          ))}

          {activeTab === "accessories" && accessories.map((item, index) => (
            <Card 
              key={index}
              className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-6 hover:shadow-glow transition-all duration-300 animate-scale-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="text-xs text-primary mb-2">{item.category}</div>
              <h3 className="text-xl font-bold mb-4">{item.name}</h3>
              <p className="text-2xl font-bold text-accent mb-4">{item.price}</p>
              <Button className="w-full bg-accent hover:bg-accent/90 text-accent-foreground">
                Add to Cart
              </Button>
            </Card>
          ))}

          {activeTab === "precautions" && precautions.map((item, index) => (
            <Card 
              key={index}
              className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-6 hover:shadow-glow transition-all duration-300 animate-scale-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="w-12 h-12 bg-primary/20 rounded-full flex items-center justify-center mb-4">
                <AlertCircle className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-3">{item.title}</h3>
              <p className="text-muted-foreground">{item.description}</p>
            </Card>
          ))}
        </div>
      </main>
    </div>
  );
};

export default Veterinary;
