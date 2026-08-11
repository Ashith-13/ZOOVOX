import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Camera, Waves, ScanFace } from "lucide-react";
import animalsBackground from "@/assets/animals-background.jpg";

const Auth = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [showFaceRecognition, setShowFaceRecognition] = useState(false);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate authentication
    setTimeout(() => {
      setIsLoading(false);
      toast.success("Credentials verified! Starting face recognition...");
      setShowFaceRecognition(true);
    }, 1500);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user' } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      toast.error("Camera access denied. Please allow camera access.");
      console.error("Camera error:", error);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  const performFaceRecognition = () => {
    setIsRecognizing(true);
    
    // Simulate face recognition process
    setTimeout(() => {
      toast.success("Face recognized successfully!");
      setTimeout(() => {
        stopCamera();
        setShowFaceRecognition(false);
        navigate("/dashboard");
        setIsRecognizing(false);
      }, 1000);
    }, 3000);
  };

  useEffect(() => {
    if (showFaceRecognition) {
      startCamera();
    }
    return () => {
      stopCamera();
    };
  }, [showFaceRecognition]);

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate registration
    setTimeout(() => {
      setIsLoading(false);
      toast.success("Account created! Setting up face recognition...");
      setShowFaceRecognition(true);
    }, 1500);
  };

  const handleFaceRecognition = () => {
    setShowFaceRecognition(true);
  };

  return (
    <div className="min-h-screen relative overflow-hidden flex items-center justify-center p-4">
      {/* Animated background */}
      <div 
        className="absolute inset-0 bg-cover bg-center opacity-20 animate-float"
        style={{ backgroundImage: `url(${animalsBackground})` }}
      />
      <div className="absolute inset-0 bg-gradient-hero" />
      
      {/* Floating orbs */}
      <div className="absolute top-20 left-20 w-64 h-64 bg-primary/20 rounded-full blur-3xl animate-glow" />
      <div className="absolute bottom-20 right-20 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-glow" style={{ animationDelay: "1s" }} />

      {/* Content */}
      <div className="relative z-10 w-full max-w-md animate-fade-in">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Waves className="w-12 h-12 text-primary animate-pulse" />
          </div>
          <h1 className="text-5xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-2">
            ZOOVOX
          </h1>
          <p className="text-xl text-muted-foreground">Speak Beyond Species</p>
        </div>

        <Card className="backdrop-blur-xl bg-card/40 border-border/50 shadow-card p-6 animate-scale-in">
          <Tabs defaultValue="signin" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="signin">Sign In</TabsTrigger>
              <TabsTrigger value="signup">Sign Up</TabsTrigger>
            </TabsList>

            <TabsContent value="signin">
              <form onSubmit={handleSignIn} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="your@email.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="bg-background/50"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="bg-background/50"
                  />
                </div>
                <Button 
                  type="submit" 
                  className="w-full bg-gradient-primary hover:opacity-90 transition-opacity"
                  disabled={isLoading}
                >
                  {isLoading ? "Signing in..." : "Sign In"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  className="w-full border-primary/50 hover:bg-primary/10"
                  onClick={handleFaceRecognition}
                >
                  <Camera className="w-4 h-4 mr-2" />
                  Sign in with Face Recognition
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="signup">
              <form onSubmit={handleSignUp} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <Input
                    id="name"
                    type="text"
                    placeholder="John Doe"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    className="bg-background/50"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="signup-email">Email</Label>
                  <Input
                    id="signup-email"
                    type="email"
                    placeholder="your@email.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="bg-background/50"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="signup-password">Password</Label>
                  <Input
                    id="signup-password"
                    type="password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="bg-background/50"
                  />
                </div>
                <Button 
                  type="submit" 
                  className="w-full bg-gradient-primary hover:opacity-90 transition-opacity"
                  disabled={isLoading}
                >
                  {isLoading ? "Creating account..." : "Sign Up"}
                </Button>
                <p className="text-xs text-muted-foreground text-center">
                  Face recognition will be set up after registration
                </p>
              </form>
            </TabsContent>
          </Tabs>
        </Card>
      </div>

      {/* Face Recognition Modal */}
      <Dialog open={showFaceRecognition} onOpenChange={(open) => {
        if (!open && !isRecognizing) {
          stopCamera();
          setShowFaceRecognition(false);
        }
      }}>
        <DialogContent className="sm:max-w-md backdrop-blur-xl bg-card/90 border-border/50">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              <ScanFace className="w-6 h-6 text-primary animate-pulse" />
              Face Recognition
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              
              {/* Scanning overlay */}
              {isRecognizing && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/50">
                  <div className="relative">
                    <div className="w-48 h-48 border-4 border-primary rounded-full animate-ping opacity-75" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <ScanFace className="w-24 h-24 text-primary animate-pulse" />
                    </div>
                  </div>
                </div>
              )}
              
              {/* Face frame guide */}
              {!isRecognizing && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-48 h-64 border-4 border-primary/50 rounded-3xl" />
                </div>
              )}
            </div>

            <div className="text-center text-sm text-muted-foreground">
              {isRecognizing ? (
                <p className="animate-pulse">Recognizing your face...</p>
              ) : (
                <p>Position your face within the frame</p>
              )}
            </div>

            <Button
              onClick={performFaceRecognition}
              disabled={isRecognizing}
              className="w-full bg-gradient-primary hover:opacity-90 transition-opacity"
            >
              {isRecognizing ? "Recognizing..." : "Start Recognition"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Auth;