import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Mic, Volume2, PawPrint, User, MapPin, Menu, LogOut } from "lucide-react";
import { toast } from "sonner";
import { useNavigate } from "react-router-dom";
import voiceWaves from "@/assets/voice-waves.jpg";
import animalsBackground from "@/assets/animals-background.jpg";

const Dashboard = () => {
  const navigate = useNavigate();
  const [isRecordingAnimal, setIsRecordingAnimal] = useState(false);
  const [isRecordingHuman, setIsRecordingHuman] = useState(false);
  const [showMenu, setShowMenu] = useState(false);

  const handleAnimalVoice = () => {
    setIsRecordingAnimal(!isRecordingAnimal);
    if (!isRecordingAnimal) {
      toast.success("Listening to animal sounds...");
    } else {
      toast.info("Translation: 'I'm happy to see you!'");
    }
  };

  const handleHumanVoice = () => {
    setIsRecordingHuman(!isRecordingHuman);
    if (!isRecordingHuman) {
      toast.success("Recording your voice...");
    } else {
      toast.info("Translating to animal language...");
    }
  };

  const handleLogout = () => {
    toast.success("Logged out successfully");
    navigate("/");
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated background */}
      <div 
        className="absolute inset-0 bg-cover bg-center opacity-10 animate-float"
        style={{ backgroundImage: `url(${animalsBackground})` }}
      />
      <div className="absolute inset-0 bg-background" />
      
      {/* Floating orbs */}
      <div className="absolute top-40 right-40 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-glow" />
      <div className="absolute bottom-20 left-20 w-80 h-80 bg-accent/10 rounded-full blur-3xl animate-glow" style={{ animationDelay: "2s" }} />

      {/* Header */}
      <header className="relative z-20 border-b border-border/50 backdrop-blur-lg bg-card/30">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-primary rounded-lg flex items-center justify-center">
              <PawPrint className="w-6 h-6 text-background" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                ZOOVOX
              </h1>
              <p className="text-xs text-muted-foreground">Speak Beyond Species</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigate("/veterinary")}
              className="hidden md:flex border-primary/50 hover:bg-primary/10"
            >
              <MapPin className="w-4 h-4 mr-2" />
              Veterinary Care
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowMenu(!showMenu)}
              className="md:hidden"
            >
              <Menu className="w-5 h-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleLogout}
              className="hover:bg-destructive/10 hover:text-destructive"
            >
              <LogOut className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </header>

      {/* Mobile menu */}
      {showMenu && (
        <div className="relative z-30 md:hidden">
          <Card className="mx-4 mt-2 p-4 backdrop-blur-xl bg-card/90 border-border/50 animate-scale-in">
            <Button
              variant="outline"
              className="w-full border-primary/50 hover:bg-primary/10"
              onClick={() => {
                navigate("/veterinary");
                setShowMenu(false);
              }}
            >
              <MapPin className="w-4 h-4 mr-2" />
              Veterinary Care
            </Button>
          </Card>
        </div>
      )}

      {/* Main content */}
      <main className="relative z-10 container mx-auto px-4 py-12">
        <div className="text-center mb-12 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
            Voice Translation Hub
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Bridge the communication gap between humans and animals with AI-powered real-time translation
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Animal to Human */}
          <Card className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-8 hover:shadow-glow transition-all duration-300 animate-scale-in group">
            <div 
              className="w-full h-48 mb-6 rounded-lg bg-cover bg-center opacity-60 group-hover:opacity-80 transition-opacity"
              style={{ backgroundImage: `url(${voiceWaves})` }}
            />
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-primary/20 rounded-full flex items-center justify-center">
                <PawPrint className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-2xl font-bold">Animal → Human</h3>
            </div>
            <p className="text-muted-foreground mb-6">
              Capture animal sounds and translate them into human-understandable speech in real-time
            </p>
            <Button
              onClick={handleAnimalVoice}
              className={`w-full ${
                isRecordingAnimal
                  ? "bg-destructive hover:bg-destructive/90"
                  : "bg-gradient-primary hover:opacity-90"
              } transition-all`}
              size="lg"
            >
              {isRecordingAnimal ? (
                <>
                  <Volume2 className="w-5 h-5 mr-2 animate-pulse" />
                  Stop Listening
                </>
              ) : (
                <>
                  <Mic className="w-5 h-5 mr-2" />
                  Start Listening
                </>
              )}
            </Button>
          </Card>

          {/* Human to Animal */}
          <Card className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-8 hover:shadow-glow transition-all duration-300 animate-scale-in group" style={{ animationDelay: "0.1s" }}>
            <div 
              className="w-full h-48 mb-6 rounded-lg bg-cover bg-center opacity-60 group-hover:opacity-80 transition-opacity"
              style={{ backgroundImage: `url(${voiceWaves})` }}
            />
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-accent/20 rounded-full flex items-center justify-center">
                <User className="w-6 h-6 text-accent" />
              </div>
              <h3 className="text-2xl font-bold">Human → Animal</h3>
            </div>
            <p className="text-muted-foreground mb-6">
              Speak naturally and translate your voice into animal-understandable communication
            </p>
            <Button
              onClick={handleHumanVoice}
              className={`w-full ${
                isRecordingHuman
                  ? "bg-destructive hover:bg-destructive/90"
                  : "bg-accent hover:bg-accent/90 text-accent-foreground"
              } transition-all`}
              size="lg"
            >
              {isRecordingHuman ? (
                <>
                  <Volume2 className="w-5 h-5 mr-2 animate-pulse" />
                  Stop Recording
                </>
              ) : (
                <>
                  <Mic className="w-5 h-5 mr-2" />
                  Start Recording
                </>
              )}
            </Button>
          </Card>
        </div>

        {/* Features preview */}
        <div className="mt-16 text-center animate-fade-in" style={{ animationDelay: "0.2s" }}>
          <h3 className="text-2xl font-bold mb-4">Additional Features</h3>
          <div className="flex flex-wrap justify-center gap-4">
            <Button
              variant="outline"
              onClick={() => navigate("/veterinary")}
              className="border-primary/50 hover:bg-primary/10"
            >
              <MapPin className="w-4 h-4 mr-2" />
              Find Veterinary Care
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
